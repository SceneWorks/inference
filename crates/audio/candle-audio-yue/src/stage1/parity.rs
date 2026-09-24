//! Stage-1 parity against the Python reference (sc-19380).
//!
//! The fixture `tests/fixtures/yue_stage1_parity.json` is written by
//! `scripts/reference/yue_stage1_reference.py`: YuE-v1 `infer.py`'s per-segment Hugging Face
//! `generate` call (CFG 1.5 → 1.2, top-p 0.93, the library-default top-k 50, repetition penalty 1.1,
//! min 100 new tokens, forced `<EOA>`) plus the epic's allow-list and smart context, on a tiny
//! `LlamaForCausalLM` whose weights are a pure function of the tensor name — rebuilt here
//! bit-identically ([`tiny_weights`]) — with the categorical draw made from candle-llm's SplitMix64
//! stream on both sides. The token streams must match exactly.
//!
//! **Noise floor.** The two sides compute the same float32 forward with different kernels (torch vs
//! candle CPU, and full re-prefill per segment vs this engine's incremental KV cache). Measured on
//! the fixture's first-draw top-64 scores (2026-09-24, CPU): max |Δ| = 1.0e-5 with CFG (mixed
//! log-probabilities, mostly the shared log-normalizer) and 3.8e-6 without (raw logits of magnitude
//! ~8, a few ulp). A token stream can only diverge when a draw lands within that distance of a
//! top-k / top-p / inverse-CDF boundary — a per-draw chance of order 1e-4 at worst over the
//! fixture's ~870 draws — so the streams are asserted equal, not within a tolerance.

use std::collections::HashMap;
use std::path::Path;

use candle_audio::candle_core::{Device, Tensor};
use candle_audio::gen_core::{self, CancelFlag};
use candle_llm::primitives::sampler::SplitMix64;
use candle_llm::primitives::weights::Weights;
use candle_llm::{CausalLm, ModelConfig};
use serde_json::Value;

use super::{SegmentStart, Stage1Lm, Stage1Model, Stage1Step};
use crate::config::{Assets, DecodeConfig, Guidance, Tier, Variant, YueRequest};
use crate::engine::{YueEngine, YueEvent};
use crate::stages::StageSet;
use crate::tokenizer::{PromptInput, PromptTokenizer, SegmentPrompt, Stage1Prompt};
use crate::tokens::{split_raw_output, CODEC_OFFSET, EOA};

/// Max |Δ| allowed between this engine's and the reference's first-draw scores (log-probabilities
/// under CFG, raw logits without): 10× the measured floor (1.0e-5). A real defect (a wrong CFG
/// scale, an unmasked unconditional stream) moves them by orders of magnitude more.
const SCORE_TOLERANCE: f32 = 1e-4;

fn fixture() -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/yue_stage1_parity.json");
    serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap()
}

fn ids(v: &Value) -> Vec<u32> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u32)
        .collect()
}

fn fnv1a64(name: &str) -> u64 {
    name.bytes().fold(0xCBF2_9CE4_8422_2325, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01B3)
    })
}

/// The reference script's weight generator, bit-identically: `(2u - 1) * scale` in f32 with `u`
/// the SplitMix64 stream seeded by the tensor name's FNV-1a-64, norms all ones, then the bias
/// channel. Returns `(name, shape, row-major values)`.
fn tiny_weights(fx: &Value) -> Vec<(String, Vec<usize>, Vec<f32>)> {
    let cfg = &fx["config"];
    let n = |k: &str| cfg[k].as_u64().unwrap() as usize;
    let (h, inter, vocab) = (n("hidden_size"), n("intermediate_size"), n("vocab_size"));
    let kv = n("num_key_value_heads") * (h / n("num_attention_heads"));
    let scale = |k: &str| fx["scales"][k].as_f64().unwrap() as f32;
    let bias = |k: &str| fx["bias"][k].as_f64().unwrap() as f32;
    let mut specs: Vec<(String, Vec<usize>, &str)> =
        vec![("model.embed_tokens.weight".into(), vec![vocab, h], "embed")];
    for l in 0..n("num_hidden_layers") {
        let p = format!("model.layers.{l}");
        specs.extend([
            (format!("{p}.input_layernorm.weight"), vec![h], "norm"),
            (format!("{p}.self_attn.q_proj.weight"), vec![h, h], "attn"),
            (format!("{p}.self_attn.k_proj.weight"), vec![kv, h], "attn"),
            (format!("{p}.self_attn.v_proj.weight"), vec![kv, h], "attn"),
            (format!("{p}.self_attn.o_proj.weight"), vec![h, h], "attn"),
            (
                format!("{p}.post_attention_layernorm.weight"),
                vec![h],
                "norm",
            ),
            (format!("{p}.mlp.gate_proj.weight"), vec![inter, h], "mlp"),
            (format!("{p}.mlp.up_proj.weight"), vec![inter, h], "mlp"),
            (format!("{p}.mlp.down_proj.weight"), vec![h, inter], "mlp"),
        ]);
    }
    specs.push(("model.norm.weight".into(), vec![h], "norm"));
    specs.push(("lm_head.weight".into(), vec![vocab, h], "lm_head"));
    specs
        .into_iter()
        .map(|(name, shape, kind)| {
            let len: usize = shape.iter().product();
            let mut w: Vec<f32> = if kind == "norm" {
                vec![1.0; len]
            } else {
                let s = scale(kind);
                let mut rng = SplitMix64::new(fnv1a64(&name));
                (0..len)
                    .map(|_| {
                        use candle_llm::primitives::sampler::TokenRng;
                        (rng.next_f32() * 2.0 - 1.0) * s
                    })
                    .collect()
            };
            let cols = *shape.last().unwrap();
            if name == "model.embed_tokens.weight" {
                w.iter_mut()
                    .step_by(cols)
                    .for_each(|x| *x = bias("embedChannel"));
            } else if name.ends_with("o_proj.weight") || name.ends_with("down_proj.weight") {
                w[..cols].iter_mut().for_each(|x| *x = 0.0);
            } else if name == "lm_head.weight" {
                w.iter_mut().step_by(cols).for_each(|x| *x = 0.0);
                for row in CODEC_OFFSET as usize..CODEC_OFFSET as usize + 1024 {
                    w[row * cols] = bias("cb0");
                }
                w[EOA as usize * cols] = bias("eoa");
            }
            (name, shape, w)
        })
        .collect()
}

fn tiny_tensors(fx: &Value, device: &Device) -> HashMap<String, Tensor> {
    tiny_weights(fx)
        .into_iter()
        .map(|(name, shape, w)| (name, Tensor::from_vec(w, shape, device).unwrap()))
        .collect()
}

/// The tiny reference model on the CPU (f32 compute).
fn tiny_model(fx: &Value) -> CausalLm {
    let device = Device::Cpu;
    let weights = Weights::from_map(tiny_tensors(fx, &device), device);
    let cfg = ModelConfig::from_json(&fx["config"]).unwrap();
    CausalLm::from_weights_format(&weights, "", cfg, None).unwrap()
}

/// Stage `fx`'s tiny model as a production snapshot directory (config, weights, a word-level
/// tokenizer.json the provider requires) under `dir`.
pub(crate) fn write_tiny_snapshot(fx: &Value, dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("config.json"), fx["config"].to_string()).unwrap();
    candle_audio::candle_core::safetensors::save(
        &tiny_tensors(fx, &Device::Cpu),
        dir.join("model.safetensors"),
    )
    .unwrap();
    let tokenizer = serde_json::json!({
        "version": "1.0",
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": null,
        "decoder": null,
        "model": {"type": "WordLevel", "vocab": {"<unk>": 0, "<s>": 1, "</s>": 2}, "unk_token": "<unk>"}
    });
    std::fs::write(dir.join("tokenizer.json"), tokenizer.to_string()).unwrap();
}

fn decode_config(fx: &Value, guided: bool) -> DecodeConfig {
    let d = &fx["decode"];
    let f = |k: &str| d[k].as_f64().unwrap() as f32;
    let u = |k: &str| d[k].as_u64().unwrap() as u32;
    DecodeConfig {
        max_new_tokens: u("maxNewTokens"),
        min_new_tokens: u("minNewTokens"),
        repetition_penalty: f("repetitionPenalty"),
        top_p: f("topP"),
        top_k: u("topK"),
        temperature: f("temperature"),
        guidance: if guided {
            Guidance::On {
                first: d["guidance"][0].as_f64().unwrap() as f32,
                rest: d["guidance"][1].as_f64().unwrap() as f32,
            }
        } else {
            Guidance::Off
        },
    }
}

/// What one direct-driven render produced.
struct Render {
    /// Per segment: the generated tokens with the closing `<EOA>` (sampled or forced).
    segments: Vec<Vec<u32>>,
    ended_by: Vec<&'static str>,
    sequence: Vec<u32>,
    first_scores: Vec<Vec<f32>>,
}

/// Drive `lm` over `prompts` with the engine's loop shape (budgeted `step`s, then `end_segment`).
fn render(lm: &mut Stage1Lm, prompts: &[Vec<u32>], decode: &DecodeConfig, seed: u64) -> Render {
    lm.begin_render(seed).unwrap();
    let (mut segments, mut ended_by) = (Vec::new(), Vec::new());
    for (index, prompt) in prompts.iter().enumerate() {
        lm.begin_segment(&SegmentStart {
            index,
            prompt,
            guidance_scale: decode.guidance.scale_for(index),
            decode,
        })
        .unwrap();
        let mut tokens = Vec::new();
        let mut how = "budget";
        while tokens.len() < decode.max_new_tokens as usize {
            match lm.step().unwrap() {
                Stage1Step::Token(t) => tokens.push(t),
                Stage1Step::EndOfAudio => {
                    how = "eoa";
                    break;
                }
            }
        }
        lm.end_segment().unwrap();
        tokens.push(EOA);
        segments.push(tokens);
        ended_by.push(how);
    }
    Render {
        segments,
        ended_by,
        sequence: lm.sequence().to_vec(),
        first_scores: lm.first_scores.clone(),
    }
}

fn prompts(fx: &Value) -> Vec<Vec<u32>> {
    fx["prompts"].as_array().unwrap().iter().map(ids).collect()
}

fn seed(fx: &Value) -> u64 {
    fx["decode"]["seed"].as_u64().unwrap()
}

/// Max |Δ| between `ours` and the reference's top-64 first-draw scores, per segment.
fn score_gap(ours: &[Vec<f32>], reference: &Value) -> f32 {
    let reference = reference.as_array().unwrap();
    assert_eq!(
        ours.len(),
        reference.len(),
        "one first-draw row per segment"
    );
    let mut gap = 0.0f32;
    for (row, top) in ours.iter().zip(reference) {
        for pair in top.as_array().unwrap() {
            let id = pair[0].as_u64().unwrap() as usize;
            let want = pair[1].as_f64().unwrap() as f32;
            gap = gap.max((row[id] - want).abs());
        }
    }
    gap
}

fn check_run(run_key: &str, guided: bool) {
    let fx = fixture();
    let run = &fx["runs"][run_key];
    let mut lm = Stage1Lm::from_model(tiny_model(&fx)).unwrap();
    let got = render(
        &mut lm,
        &prompts(&fx),
        &decode_config(&fx, guided),
        seed(&fx),
    );

    let gap = score_gap(&got.first_scores, &run["firstScoresTop64"]);
    eprintln!("{run_key}: first-draw score max |Δ| = {gap:e}");
    assert!(gap <= SCORE_TOLERANCE, "{run_key}: score gap {gap:e}");

    let want: Vec<Vec<u32>> = run["segments"]
        .as_array()
        .unwrap()
        .iter()
        .map(ids)
        .collect();
    for (i, (g, w)) in got.segments.iter().zip(&want).enumerate() {
        assert_eq!(g, w, "{run_key}: segment {i} token stream");
    }
    assert_eq!(got.segments.len(), want.len());
    let ended: Vec<&str> = run["endedBy"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    assert_eq!(got.ended_by, ended, "{run_key}: segment endings");
    assert_eq!(got.sequence, ids(&run["sequence"]), "{run_key}: raw output");

    let split = split_raw_output(&got.sequence, 0).unwrap();
    assert_eq!(
        split.vocals,
        ids(&run["split"]["vocals"]),
        "{run_key}: vocals"
    );
    assert_eq!(
        split.instrumental,
        ids(&run["split"]["instrumental"]),
        "{run_key}: instrumental"
    );
}

#[test]
fn guided_render_matches_the_reference_token_for_token() {
    check_run("guidance", true);
}

#[test]
fn unguided_render_matches_the_reference_token_for_token() {
    check_run("noGuidance", false);
}

#[test]
fn the_fixture_exercises_both_endings_the_smart_context_and_the_min_new_floor() {
    let fx = fixture();
    let min_new = fx["decode"]["minNewTokens"].as_u64().unwrap() as usize;
    let max_new = fx["decode"]["maxNewTokens"].as_u64().unwrap() as usize;
    let mut endings = Vec::new();
    for run in fx["runs"].as_object().unwrap().values() {
        for (seg, how) in run["segments"]
            .as_array()
            .unwrap()
            .iter()
            .zip(run["endedBy"].as_array().unwrap())
        {
            let n = seg.as_array().unwrap().len() - 1; // tokens before the closing <EOA>
            assert!(n >= min_new, "no segment ends before the floor");
            endings.push((how.as_str().unwrap().to_string(), n == max_new));
        }
        // The last segment's window was shortened (the sequence outgrew the context budget).
        let windows = run["windowLengths"].as_array().unwrap();
        let prompt_total: usize = fx["prompts"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| p.as_array().unwrap().len())
            .sum();
        let full = prompt_total
            + run["segments"]
                .as_array()
                .unwrap()
                .iter()
                .take(windows.len() - 1)
                .map(|s| s.as_array().unwrap().len())
                .sum::<usize>();
        assert!(windows.last().unwrap().as_u64().unwrap() < full as u64);
    }
    assert!(endings.contains(&("eoa".into(), false)), "{endings:?}");
    assert!(endings.contains(&("budget".into(), true)), "{endings:?}");
}

/// The golden depends on each knob: flipping one (the CFG schedule, the min-new floor, the
/// repetition penalty, top-k) changes the stream. Guards against a golden the engine would match
/// with a knob silently ignored.
#[test]
fn each_decode_knob_moves_the_stream() {
    let fx = fixture();
    let want: Vec<Vec<u32>> = fx["runs"]["guidance"]["segments"]
        .as_array()
        .unwrap()
        .iter()
        .map(ids)
        .collect();
    let base = decode_config(&fx, true);
    let variants = [
        DecodeConfig {
            guidance: Guidance::On {
                first: 1.2,
                rest: 1.5,
            },
            ..base
        },
        DecodeConfig {
            min_new_tokens: 0,
            ..base
        },
        DecodeConfig {
            repetition_penalty: 1.0,
            ..base
        },
        DecodeConfig { top_k: 0, ..base },
    ];
    let mut lm = Stage1Lm::from_model(tiny_model(&fx)).unwrap();
    for (i, d) in variants.iter().enumerate() {
        let got = render(&mut lm, &prompts(&fx), d, seed(&fx));
        assert_ne!(got.segments, want, "variant {i} must change the stream");
    }
    // …and the seed is the render's only randomness.
    let again = render(&mut lm, &prompts(&fx), &base, seed(&fx));
    assert_eq!(again.segments, want);
}

#[test]
fn smart_context_matches_the_reference_shorten_input() {
    for case in fixture()["shortenCases"].as_array().unwrap() {
        let got = super::shorten_context(
            &ids(&case["seq"]),
            case["maxContext"].as_u64().unwrap() as usize,
        );
        assert_eq!(got, ids(&case["out"]), "{}", case["name"]);
    }
}

#[test]
fn split_matches_the_reference_save() {
    for case in fixture()["splitCases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let got = split_raw_output(
            &ids(&case["raw"]),
            case["skipPairs"].as_u64().unwrap() as usize,
        );
        match case.get("error") {
            Some(e) => assert!(got.is_err(), "{name}: the reference refuses ({e})"),
            None => {
                let got = got.unwrap_or_else(|e| panic!("{name}: {e}"));
                assert_eq!(got.vocals, ids(&case["vocals"]), "{name}: vocals");
                assert_eq!(got.instrumental, ids(&case["instrumental"]), "{name}");
            }
        }
    }
}

#[test]
fn tiny_weights_match_the_reference_generator() {
    let fx = fixture();
    for (name, _, w) in tiny_weights(&fx) {
        let want = fx["tensorSums"][&name].as_f64().unwrap();
        let got: f64 = w.iter().map(|&x| f64::from(x)).sum();
        assert!(
            (got - want).abs() <= 1e-9 * want.abs().max(1.0),
            "{name}: sum {got} vs reference {want}"
        );
    }
}

/// Production wiring: `stage1::load` (the `StageSet::production` loader) loads a staged snapshot
/// through candle-llm's provider — no `Unsupported` refusal — and decodes inside the allow-list.
#[test]
fn production_load_decodes_a_staged_snapshot() {
    let fx = fixture();
    let root = tempfile::tempdir().unwrap();
    write_tiny_snapshot(&fx, root.path());
    let assets = Assets {
        stage1: root.path().into(),
        stage2: "/unused".into(),
        xcodec: "/unused".into(),
    };
    let mut lm = (StageSet::production().stage1)(&assets, None).expect("stage 1 loads");
    let decode = decode_config(&fx, true);
    lm.begin_render(7).unwrap();
    lm.begin_segment(&SegmentStart {
        index: 0,
        prompt: &prompts(&fx)[0],
        guidance_scale: decode.guidance.scale_for(0),
        decode: &decode,
    })
    .unwrap();
    for _ in 0..8 {
        match lm.step().unwrap() {
            Stage1Step::Token(t) => {
                assert!((CODEC_OFFSET..=crate::tokens::STAGE1_ALLOW_MAX).contains(&t))
            }
            Stage1Step::EndOfAudio => panic!("<EOA> before the min-new-tokens floor"),
        }
    }
    lm.end_segment().unwrap();
}

#[test]
fn tiered_repo_roots_resolve_to_the_asserted_tier() {
    let root = tempfile::tempdir().unwrap();
    for tier in ["bf16", "q8"] {
        std::fs::create_dir_all(root.path().join(tier)).unwrap();
        std::fs::write(root.path().join(tier).join("config.json"), "{}").unwrap();
    }
    let dir = |t| super::lm::lm_snapshot_dir(root.path(), t);
    assert_eq!(dir(None), root.path().join("bf16"));
    assert_eq!(dir(Some(Tier::Q8)), root.path().join("q8"));
    // An unstaged tier falls back to bf16 (quantized on load).
    assert_eq!(dir(Some(Tier::Q4)), root.path().join("bf16"));
    // A snapshot directory is used as is.
    assert_eq!(
        super::lm::lm_snapshot_dir(&root.path().join("q8"), None),
        root.path().join("q8")
    );
}

/// A prompt builder that returns the fixture's blocks, so the engine drives the reference prompts.
struct FixturePrompts(Vec<Vec<u32>>);

impl PromptTokenizer for FixturePrompts {
    fn build(&self, _: &PromptInput<'_>) -> gen_core::Result<Stage1Prompt> {
        Ok(Stage1Prompt {
            segments: self
                .0
                .iter()
                .enumerate()
                .map(|(i, ids)| SegmentPrompt {
                    label: format!("s{i}"),
                    ids: ids.clone(),
                })
                .collect(),
        })
    }
}

/// Stage 1 wrapped to count steps and trip `cancel` on the `trip_at`-th.
struct Tripping {
    inner: Stage1Lm,
    steps: std::sync::Arc<std::sync::Mutex<usize>>,
    trip: Option<(usize, CancelFlag)>,
}

impl Stage1Model for Tripping {
    fn begin_render(&mut self, seed: u64) -> gen_core::Result<()> {
        self.inner.begin_render(seed)
    }
    fn begin_segment(&mut self, s: &SegmentStart<'_>) -> gen_core::Result<()> {
        self.inner.begin_segment(s)
    }
    fn step(&mut self) -> gen_core::Result<Stage1Step> {
        let mut steps = self.steps.lock().unwrap();
        *steps += 1;
        if let Some((at, flag)) = &self.trip {
            if *steps == *at {
                flag.cancel();
            }
        }
        self.inner.step()
    }
    fn end_segment(&mut self) -> gen_core::Result<()> {
        self.inner.end_segment()
    }
}

fn engine_with_tiny_stage1(
    trip: Option<(usize, CancelFlag)>,
) -> (
    YueEngine,
    YueRequest,
    std::sync::Arc<std::sync::Mutex<usize>>,
) {
    let fx = fixture();
    let blocks = prompts(&fx);
    let steps = std::sync::Arc::new(std::sync::Mutex::new(0));
    let mut stages = StageSet::stubs();
    stages.tokenizer = std::sync::Arc::new(move |_: &Assets| {
        Ok(Box::new(FixturePrompts(blocks.clone())) as Box<dyn PromptTokenizer>)
    });
    let (s, fx2) = (steps.clone(), fx.clone());
    stages.stage1 = std::sync::Arc::new(move |_: &Assets, _| {
        Ok(Box::new(Tripping {
            inner: Stage1Lm::from_model(tiny_model(&fx2))?,
            steps: s.clone(),
            trip: trip.clone(),
        }) as Box<dyn Stage1Model>)
    });
    let assets = Assets {
        stage1: "/unused".into(),
        stage2: "/unused".into(),
        xcodec: "/unused".into(),
    };
    let variant = Variant::ALL[0];
    let mut req = YueRequest::new("pop", "[verse]\na\n[chorus]\nb\n[bridge]\nc\n[outro]\nd");
    req.segments = 4;
    req.seed = seed(&fx);
    req.decode = decode_config(&fx, true);
    (YueEngine::new(variant, assets, None, stages), req, steps)
}

/// AC2: the segment loop over the real stage 1 reports one finished-segment event per segment
/// (carrying the reference's token counts) and renders through the downstream stages.
#[test]
fn engine_reports_each_segment_once_over_the_real_stage1() {
    let fx = fixture();
    let (engine, req, _) = engine_with_tiny_stage1(None);
    let mut events = Vec::new();
    let out = engine
        .render(&req, &CancelFlag::new(), &mut |e| events.push(e))
        .unwrap();
    assert!(!out.mix.is_empty());
    let finished: Vec<usize> = events
        .iter()
        .filter_map(|e| match e {
            YueEvent::SegmentFinished { tokens, .. } => Some(*tokens),
            _ => None,
        })
        .collect();
    // The engine counts generated audio tokens: neither a sampled nor a forced <EOA> is one.
    let want: Vec<usize> = fx["runs"]["guidance"]["segments"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| s.as_array().unwrap().len() - 1)
        .collect();
    assert_eq!(finished, want);
    let started = events
        .iter()
        .filter(|e| matches!(e, YueEvent::SegmentStarted { .. }))
        .count();
    assert_eq!(started, 4);
}

/// AC2: a cancel raised mid-segment stops the real stage 1 within one decode step.
#[test]
fn cancel_stops_the_real_stage1_within_one_step() {
    let cancel = CancelFlag::new();
    let (engine, req, steps) = engine_with_tiny_stage1(Some((150, cancel.clone())));
    let err = engine.render(&req, &cancel, &mut |_| {}).unwrap_err();
    assert!(matches!(err, gen_core::Error::Canceled), "{err:?}");
    assert_eq!(
        *steps.lock().unwrap(),
        150,
        "no step after the one that saw the cancel"
    );
}
