//! Stage-2 tests that run in every lane: the upstream-golden assembly parity (mock LM), the chunk
//! schedule, the repair, cancellation, and the candle-llm wiring over a tiny synthetic Llama. The
//! real-weight parity lives in `tests/stage2_real_weights.rs`; the Metal characterisation in
//! [`metal`].

use std::collections::HashMap;
use std::path::Path;

use candle_audio::candle_core::{DType, Device, Tensor};
use candle_audio::gen_core::{self, CancelFlag};

use super::*;
use crate::config::Assets;
use crate::tokens::{CODEBOOK_SIZE, CODEC_OFFSET};

// ---------------------------------------------------------------------------------------------
// The mock LM — the Rust twin of `MockModel` in scripts/reference/yue_stage2_reference.py.
// ---------------------------------------------------------------------------------------------

const MOCK_P: u64 = 4_194_301;
const MOCK_A: u64 = 2_654_435;

/// A deterministic integer "LM": row scores are a keyed permutation of `[0, P)`, plus `2P` on codes
/// `< 16` of the codebook after the last token's, unless `key % 8 == 0`. Exact in f32 (`3P < 2²⁴`).
#[derive(Default)]
pub(crate) struct MockLm {
    rows: Vec<Vec<u32>>,
    pub(crate) steps: usize,
    /// Trip this flag once `steps` reaches the given count (the cancellation test).
    pub(crate) cancel_at: Option<(usize, CancelFlag)>,
}

fn mock_scores(history: &[u32]) -> Vec<f32> {
    let n = history.len();
    let key = (n as u64 * 1_000_003
        + u64::from(history[n - 1]) * 7_919
        + u64::from(history[n - 2]) * 104_729)
        % MOCK_P;
    let last = history[n - 1];
    let expected = (CODEC_OFFSET..CODEC_OFFSET + 7 * CODEBOOK_SIZE)
        .contains(&last)
        .then(|| (last - CODEC_OFFSET) / CODEBOOK_SIZE + 1);
    let favoured = expected
        .filter(|_| !key.is_multiple_of(8))
        .map(|e| CODEC_OFFSET + e * CODEBOOK_SIZE);
    (STAGE2_SLICE_MIN..=STAGE2_SLICE_MAX)
        .map(|v| {
            let mut s = (u64::from(v) * MOCK_A + key) % MOCK_P;
            if favoured.is_some_and(|lo| (lo..lo + 16).contains(&v)) {
                s += 2 * MOCK_P;
            }
            s as f32
        })
        .collect()
}

impl Stage2Lm for MockLm {
    fn begin(&mut self, batch: usize, _capacity: usize) -> gen_core::Result<()> {
        self.rows = vec![Vec::new(); batch];
        Ok(())
    }

    fn step(&mut self, tokens: &[Vec<u32>]) -> gen_core::Result<Vec<Vec<f32>>> {
        assert_eq!(tokens.len(), self.rows.len(), "batch changed mid-group");
        self.steps += 1;
        if let Some((at, flag)) = &self.cancel_at {
            if self.steps >= *at {
                flag.cancel();
            }
        }
        Ok(self
            .rows
            .iter_mut()
            .zip(tokens)
            .map(|(row, t)| {
                row.extend(t);
                mock_scores(row)
            })
            .collect())
    }
}

fn fixture(name: &str) -> serde_json::Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name);
    serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap()
}

fn hex_row(row: &[u32]) -> String {
    row.iter().map(|c| format!("{c:03x}")).collect()
}

/// **AC2** — upstream `stage2_inference` (verbatim, mock-driven) over three schedules: multi-row
/// groups + a partial last group + a ragged tail (937 frames, batch 2), one full chunk + tail
/// (337, batch 4), and a sub-chunk track (45). Every code of all 8 codebooks, and the number of
/// codes `fix_output` repaired, must match.
#[test]
fn upstream_chunking_batching_and_repair_reassemble_identically() {
    let golden = fixture("stage2_mock_reference.json");
    assert_eq!(golden["mock"]["p"], MOCK_P);
    assert_eq!(golden["mock"]["a"], MOCK_A);
    let cases = golden["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 3);
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let frames = case["frames"].as_u64().unwrap() as u32;
        let seed = case["cb0_seed"].as_u64().unwrap() as u32;
        let batch = case["batch_size"].as_u64().unwrap() as usize;
        let cb0: Vec<u32> = (0..frames)
            .map(|t| (t * 37 + (t * t) % 101 + seed) % CODEBOOK_SIZE)
            .collect();
        let (grid, repaired) =
            upsample_with(&mut MockLm::default(), &cb0, batch, &CancelFlag::new()).unwrap();
        grid.check_against(&cb0).unwrap();
        let want: Vec<&str> = case["codebooks_hex"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r.as_str().unwrap())
            .collect();
        for (k, (row, want)) in grid.codebooks.iter().zip(&want).enumerate() {
            let got = hex_row(row);
            if got != *want {
                let t = (0..row.len())
                    .find(|&t| got[3 * t..3 * t + 3] != want[3 * t..3 * t + 3])
                    .unwrap();
                panic!(
                    "{name}: codebook {k} differs first at frame {t} (got {:?}, want {:?})",
                    &got[3 * t..3 * t + 3],
                    &want[3 * t..3 * t + 3]
                );
            }
        }
        assert_eq!(
            repaired as u64,
            case["repaired_codes"].as_u64().unwrap(),
            "{name}"
        );
        assert!(repaired > 0, "{name}: the case must exercise the repair");
    }
}

#[test]
fn chunk_plan_groups_full_chunks_then_the_ragged_tail() {
    let g = |start, rows, frames| ChunkGroup {
        start,
        rows,
        frames,
    };
    assert_eq!(
        chunk_plan(937, 2),
        [g(0, 2, 300), g(600, 1, 300), g(900, 1, 37)]
    );
    assert_eq!(chunk_plan(1200, 4), [g(0, 4, 300)]);
    assert_eq!(chunk_plan(1500, 4), [g(0, 4, 300), g(1200, 1, 300)]);
    assert_eq!(chunk_plan(45, 4), [g(0, 1, 45)]);
    assert_eq!(chunk_plan(300, 4), [g(0, 1, 300)]);
    assert!(chunk_plan(0, 4).is_empty());
    assert_eq!(
        chunk_plan(601, 0),
        [g(0, 1, 300), g(300, 1, 300), g(600, 1, 1)]
    );
}

#[test]
fn fix_output_uses_the_first_seen_most_frequent_unrepaired_value() {
    // Row 1: 5 and -3 both occur twice; 5 is seen first, so both invalid codes become 5.
    // Row 2: the invalid value 1500 dominates, so upstream "repairs" to it — the port then refuses.
    let grid = vec![
        vec![1, 2, 3],
        vec![5, -3, 7, -3, 5, 2000],
        vec![1500, 1500, 4],
    ];
    let (fixed, n) = fix_output(&grid);
    assert_eq!(fixed[0], [1, 2, 3]);
    assert_eq!(fixed[1], [5, 5, 7, 5, 5, 5]);
    assert_eq!(fixed[2], [1500, 1500, 4]);
    assert_eq!(n, 5);
}

#[test]
fn argmax_breaks_ties_to_the_lowest_id_and_refuses_nan() {
    assert_eq!(argmax(&[1.0, 3.0, 3.0, 2.0]), Some(1));
    assert_eq!(argmax(&[f32::NEG_INFINITY, f32::NEG_INFINITY]), Some(0));
    assert_eq!(argmax(&[1.0, f32::NAN]), None);
    assert_eq!(argmax(&[]), None);
}

#[test]
fn a_repair_that_leaves_an_invalid_code_is_refused() {
    /// Always picks the last slice id (codebook 7): row 1 is all-invalid, so nothing can repair it.
    struct Stray(usize);
    impl Stage2Lm for Stray {
        fn begin(&mut self, batch: usize, _: usize) -> gen_core::Result<()> {
            self.0 = batch;
            Ok(())
        }
        fn step(&mut self, _: &[Vec<u32>]) -> gen_core::Result<Vec<Vec<f32>>> {
            let mut row = vec![0.0; SLICE_LEN];
            row[SLICE_LEN - 1] = 1.0;
            Ok(vec![row; self.0])
        }
    }
    let err = upsample_with(&mut Stray(0), &[1, 2, 3], 4, &CancelFlag::new()).unwrap_err();
    assert!(err.to_string().contains("repair left code"), "{err}");
}

#[test]
fn inputs_outside_codebook_zero_or_empty_are_refused() {
    let lm = &mut MockLm::default();
    assert!(upsample_with(lm, &[], 4, &CancelFlag::new()).is_err());
    assert!(upsample_with(lm, &[1, CODEBOOK_SIZE], 4, &CancelFlag::new()).is_err());
}

/// **R6** — a cancel raised mid-decode lands before the very next LM step.
#[test]
fn cancel_lands_before_the_next_lm_step() {
    let cancel = CancelFlag::new();
    let mut lm = MockLm {
        cancel_at: Some((10, cancel.clone())),
        ..MockLm::default()
    };
    let err = upsample_with(&mut lm, &[1; 700], 2, &cancel).unwrap_err();
    assert!(matches!(err, gen_core::Error::Canceled), "{err}");
    assert_eq!(lm.steps, 10, "no step may run after the cancel");
}

// ---------------------------------------------------------------------------------------------
// A tiny synthetic Llama through candle-llm.
// ---------------------------------------------------------------------------------------------

/// The real stage-2 vocabulary width (the slice needs `>= 53526`).
const VOCAB: usize = 83_840;
const HIDDEN: usize = 32;
const LAYERS: usize = 2;

fn synthetic_config() -> serde_json::Value {
    serde_json::json!({
        "architectures": ["LlamaForCausalLM"], "model_type": "llama",
        "hidden_size": HIDDEN, "intermediate_size": 64, "num_attention_heads": 4,
        "num_hidden_layers": LAYERS, "num_key_value_heads": 2, "vocab_size": VOCAB,
        "rms_norm_eps": 1e-5, "rope_theta": 10000.0, "max_position_embeddings": 8192,
        "tie_word_embeddings": false, "torch_dtype": "bfloat16",
        "bos_token_id": 1, "eos_token_id": 2
    })
}

/// Deterministic uniform `[-scale, scale)` values (SplitMix64).
fn uniform(seed: u64, len: usize, scale: f32) -> Vec<f32> {
    let mut s = seed;
    (0..len)
        .map(|_| {
            s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            ((z >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
        })
        .collect()
}

/// The synthetic stage-2 weights. Random, except for the one piece of structure that makes the
/// teacher-forced loop meaningful: a codebook-`j` token's embedding carries a large component on
/// hidden dimension `j`, and the LM head scores codebook `j + 1` on that dimension — so, like the
/// real checkpoint, the model mostly places each residual in its own codebook (a structureless
/// random model strays so often that `fix_output` cannot repair the grid).
fn synthetic_tensors() -> HashMap<String, Tensor> {
    let (h, kv, ff) = (HIDDEN, HIDDEN / 2, 64);
    let mut t = HashMap::new();
    let mut seed = 1;
    let mut values = |len: usize, scale: f32| {
        seed += 1;
        uniform(seed, len, scale)
    };
    let mut embed = values(VOCAB * h, 0.3);
    let mut head = values(VOCAB * h, 0.3);
    let codec = CODEC_OFFSET as usize;
    for token in codec..codec + NUM_CODEBOOKS * CODEBOOK_SIZE as usize {
        let j = (token - codec) / CODEBOOK_SIZE as usize;
        embed[token * h + j] += 3.0;
        if j > 0 {
            head[token * h + j - 1] += 2.0;
        }
    }
    let tensor = |data: Vec<f32>, shape: (usize, usize)| {
        Tensor::from_vec(data, shape, &Device::Cpu).unwrap()
    };
    t.insert(
        "model.embed_tokens.weight".into(),
        tensor(embed, (VOCAB, h)),
    );
    t.insert("lm_head.weight".into(), tensor(head, (VOCAB, h)));
    let ones = || Tensor::ones(h, DType::F32, &Device::Cpu).unwrap();
    for l in 0..LAYERS {
        let p = format!("model.layers.{l}");
        let s = 1.0 / (h as f32).sqrt();
        for (name, shape, scale) in [
            ("self_attn.q_proj", (h, h), s),
            ("self_attn.k_proj", (kv, h), s),
            ("self_attn.v_proj", (kv, h), s),
            ("self_attn.o_proj", (h, h), s),
            ("mlp.gate_proj", (ff, h), s),
            ("mlp.up_proj", (ff, h), s),
            ("mlp.down_proj", (h, ff), 1.0 / (ff as f32).sqrt()),
        ] {
            let data = values(shape.0 * shape.1, scale);
            t.insert(format!("{p}.{name}.weight"), tensor(data, shape));
        }
        t.insert(format!("{p}.input_layernorm.weight"), ones());
        t.insert(format!("{p}.post_attention_layernorm.weight"), ones());
    }
    t.insert("model.norm.weight".into(), ones());
    t
}

/// Write the synthetic model as a bf16 HF snapshot (config, tokenizer, safetensors) — the shape the
/// production loader reads.
pub(crate) fn write_synthetic_snapshot(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("config.json"), synthetic_config().to_string()).unwrap();
    std::fs::write(
        dir.join("tokenizer.json"),
        r#"{"version": "1.0", "added_tokens": [], "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"}, "post_processor": null, "decoder": null,
            "model": {"type": "WordLevel", "vocab": {"t0": 0}, "unk_token": "t0"}}"#,
    )
    .unwrap();
    let bf16: HashMap<String, Tensor> = synthetic_tensors()
        .into_iter()
        .map(|(k, v)| (k, v.to_dtype(DType::BF16).unwrap()))
        .collect();
    candle_audio::candle_core::safetensors::save(&bf16, dir.join("model.safetensors")).unwrap();
}

/// The synthetic snapshot as a [`CandleStage2Lm`] on `device`, computing in `dtype`.
pub(crate) fn synthetic_lm(dir: &Path, device: &Device, dtype: DType) -> CandleStage2Lm {
    let w = candle_llm::primitives::Weights::from_dir(dir, device).unwrap();
    let cfg = candle_llm::ModelConfig::from_dir(dir).unwrap();
    CandleStage2Lm::from_model(CausalLm::from_weights_dtype(&w, "", cfg, None, dtype).unwrap())
}

fn track(frames: u32) -> Vec<u32> {
    (0..frames).map(|t| (t * 53 + 7) % CODEBOOK_SIZE).collect()
}

/// **Production wiring** — `stage2::load` (tier resolution → `LlamaProvider::load` → the
/// teacher-forced loop) no longer refuses: over a tiered root holding a synthetic bf16 tier it
/// loads, and it upsamples a track into a valid grid whose row 0 is the input.
#[test]
fn production_load_upsamples_through_candle_llm() {
    let root = tempfile::tempdir().unwrap();
    write_synthetic_snapshot(&root.path().join("bf16"));
    let assets = Assets {
        stage1: root.path().join("absent-s1"),
        stage2: root.path().to_path_buf(),
        xcodec: root.path().join("absent-xc"),
    };
    let mut stage2 = load(&assets, None).unwrap();
    let cb0 = track(12);
    let grid = stage2.upsample(&cb0, &CancelFlag::new()).unwrap();
    grid.check_against(&cb0).unwrap();
    assert!(load(&assets, Some(Tier::Q8)).is_err(), "no q8 tier staged");
}

/// **AC2 with a real LM** — batching is output-neutral: 2 full chunks + a ragged tail decoded as
/// one 2-row group produce exactly the grid of decoding every chunk alone.
#[test]
fn batched_length_groups_equal_single_row_decoding() {
    let dir = tempfile::tempdir().unwrap();
    write_synthetic_snapshot(dir.path());
    let mut lm = synthetic_lm(dir.path(), &Device::Cpu, DType::F32);
    let cb0 = track(2 * CHUNK_FRAMES as u32 + 17);
    let cancel = CancelFlag::new();
    let (batched, _) = upsample_with(&mut lm, &cb0, 2, &cancel).unwrap();
    let (single, _) = upsample_with(&mut lm, &cb0, 1, &cancel).unwrap();
    batched.check_against(&cb0).unwrap();
    assert_eq!(batched, single);
}

/// The static-cache incremental decode computes the same logits as re-running the whole context
/// from scratch each step (upstream `generate` re-prefills every frame).
#[test]
fn incremental_decode_matches_a_full_reprefill() {
    let dir = tempfile::tempdir().unwrap();
    write_synthetic_snapshot(dir.path());
    let mut lm = synthetic_lm(dir.path(), &Device::Cpu, DType::F32);
    let cb0 = track(6);
    let d = characterise(&mut lm, &mut Reprefill::new(dir.path()), &cb0).unwrap();
    assert_eq!(d.positions, 6 * RESIDUALS);
    assert!(d.mismatches.is_empty(), "{:?}", d.mismatches);
    assert!(
        d.max_abs_logit_delta <= 1e-4 * d.max_abs_logit.max(1.0),
        "incremental vs re-prefill logits differ by {} (scale {})",
        d.max_abs_logit_delta,
        d.max_abs_logit
    );
}

/// Recomputes each step over the whole history with a fresh cache (upstream's per-frame prefill).
struct Reprefill {
    lm: CandleStage2Lm,
    history: Vec<u32>,
}

impl Reprefill {
    fn new(dir: &Path) -> Self {
        Self {
            lm: synthetic_lm(dir, &Device::Cpu, DType::F32),
            history: Vec::new(),
        }
    }
}

impl Stage2Lm for Reprefill {
    fn begin(&mut self, batch: usize, _: usize) -> gen_core::Result<()> {
        assert_eq!(batch, 1);
        self.history.clear();
        Ok(())
    }

    fn step(&mut self, tokens: &[Vec<u32>]) -> gen_core::Result<Vec<Vec<f32>>> {
        self.history.extend(&tokens[0]);
        self.lm.begin(1, self.history.len())?;
        self.lm.step(std::slice::from_ref(&self.history))
    }
}

/// The two divergence metrics the Metal bounds are stated in.
#[test]
fn divergence_metrics_are_relative_delta_and_flip_fraction() {
    let flip = Mismatch {
        frame: 0,
        codebook: 1,
        reference: STAGE2_SLICE_MIN,
        other: STAGE2_SLICE_MIN + 1,
        reference_margin: 0.1,
    };
    let d = Divergence {
        positions: 8,
        mismatches: vec![flip; 2],
        max_abs_logit_delta: 0.5,
        max_abs_logit: 20.0,
        ..Divergence::default()
    };
    assert_eq!(d.relative_logit_delta(), 0.025);
    assert_eq!(d.flip_fraction(), 0.25);
    assert_eq!(Divergence::default().relative_logit_delta(), 0.0);
    assert_eq!(Divergence::default().flip_fraction(), 0.0);
}

/// **AC3** — Metal computes in bf16 (candle-llm's GPU compute dtype, [`candle_llm::compute_dtype`]);
/// its residual picks are characterised against the CPU f32 reference on the same weights rather
/// than assumed equal. Compiled only with the `metal` feature (absent from every build without
/// it); run on Metal hardware with
/// `cargo test -p candle-audio-yue --features metal --lib stage2::tests::metal -- --nocapture`.
/// A `metal` build with no Metal device fails rather than skips. (candle's CPU backend has no bf16
/// matmul, so there is no CPU stand-in for this.)
#[cfg(feature = "metal")]
mod metal {
    use super::*;

    /// Bound on `max |Δlogit| / max |logit|` for Metal bf16 against CPU f32 on the synthetic model.
    /// PROVISIONAL until the first Metal run (sc-19381): set from the measured value with headroom.
    const MAX_REL_LOGIT_DELTA: f32 = 0.08;
    /// Bound on the fraction of residual picks that flip (teacher-forced on the reference stream).
    /// PROVISIONAL until the first Metal run (sc-19381): set from the measured value with headroom.
    const MAX_FLIP_FRACTION: f64 = 0.10;

    /// Characterise a bf16-computing LM against the f32 reference over one 24-frame chunk: report
    /// the flips and their margins, and bound both the logit noise
    /// ([`Divergence::relative_logit_delta`]) and the flip rate ([`Divergence::flip_fraction`]).
    fn check_bf16_divergence(
        tag: &str,
        reference: &mut CandleStage2Lm,
        bf16: &mut CandleStage2Lm,
    ) -> Divergence {
        let cb0 = track(24);
        let d = characterise(reference, bf16, &cb0).unwrap();
        eprintln!(
            "{tag}: {} of {} residual picks differ; max |Δlogit| {:.4} at logit scale {:.3}; \
             margins {:?}",
            d.mismatches.len(),
            d.positions,
            d.max_abs_logit_delta,
            d.max_abs_logit,
            d.mismatches
                .iter()
                .map(|m| m.reference_margin)
                .collect::<Vec<_>>()
        );
        assert_eq!(d.positions, cb0.len() * RESIDUALS);
        assert!(
            d.relative_logit_delta() <= MAX_REL_LOGIT_DELTA,
            "{tag}: relative logit delta {} exceeds {MAX_REL_LOGIT_DELTA}",
            d.relative_logit_delta()
        );
        assert!(
            d.flip_fraction() <= MAX_FLIP_FRACTION,
            "{tag}: flip fraction {} exceeds {MAX_FLIP_FRACTION}",
            d.flip_fraction()
        );
        d
    }

    #[test]
    fn metal_bf16_divergence_from_cpu_f32_is_characterised() {
        let device = Device::new_metal(0).expect("the metal feature requires a Metal device");
        let dir = tempfile::tempdir().unwrap();
        write_synthetic_snapshot(dir.path());
        let mut cpu = synthetic_lm(dir.path(), &Device::Cpu, DType::F32);
        let mut gpu = synthetic_lm(dir.path(), &device, candle_llm::compute_dtype(&device));
        assert_eq!(gpu.model().compute_dtype(), DType::BF16);
        check_bf16_divergence("metal bf16", &mut cpu, &mut gpu);
        // The GPU's own teacher-forced output is a valid grid over the same input.
        let cb0 = track(CHUNK_FRAMES as u32 + 5);
        let (grid, _) = upsample_with(&mut gpu, &cb0, 2, &CancelFlag::new()).unwrap();
        grid.check_against(&cb0).unwrap();
    }
}
