//! Real-weight loader smoke for the YuE q4 tiers produced by `prepare_snapshot` (sc-19375, epic
//! sc-19373): a q4 stage-1 (7B) and a q4 stage-2 (1B) prepared snapshot load through
//! [`LlamaProvider::load`] with a plain [`LoadSpec`] — the persisted `quantization` block, not the
//! spec, selects the quantized load — and a forward over a short YuE-shaped prompt yields finite
//! logits of the checkpoint's full vocabulary width (stage-1 83,968; stage-2 83,840).
//!
//! Gated like the other candle-llm real-weight tests: `#[ignore]`d in ordinary runs, and the
//! snapshot env var is **required** under `--ignored` (an unset var panics; it never passes
//! silently).
//!
//! ```text
//! YUE_S1_Q4_SNAPSHOT=/path/yue-s1-7b-anneal-en-cot-candle/q4 \
//! YUE_S2_Q4_SNAPSHOT=/path/yue-s2-1b-general-candle/q4 \
//!   cargo test --release -p candle-llm --test yue_prepared -- --ignored --nocapture
//! ```
//!
//! The default build (no `cuda` / `metal` feature) runs on the CPU; the 7B q4 load peaks around
//! 30 GB host RAM there.

use candle_core::{DType, Device, Tensor};
use candle_llm::LlamaProvider;
use core_llm::LoadSpec;

/// `[stage_1]`-style prompt ids from the YuE mm tokenizer: `<s>`, a short text run, `<SOA>` (32001),
/// then xcodec codebook-0 tokens inside stage-1's allow range `[45334, 56721]`.
const STAGE1_PROMPT: &[u32] = &[1, 518, 3901, 29962, 13, 32001, 45334, 46000, 50000, 56721];
/// Stage-2 teacher-forced shape: `<SOA>`, `<stage_2>` (32017), then codebook tokens inside the
/// stage-2 residual slice `[46358, 53525]`.
const STAGE2_PROMPT: &[u32] = &[32001, 32017, 45334, 46358, 47000, 50000, 53525];

fn snapshot(var: &str) -> String {
    std::env::var(var).unwrap_or_else(|_| panic!("set {var} to a prepared YuE q4 snapshot dir"))
}

/// Load `dir` through the provider's `LoadSpec` path and return the last-position logits.
fn last_logits(dir: &str, prompt: &[u32]) -> (LlamaProvider, Vec<f32>, usize) {
    let provider =
        LlamaProvider::load(&LoadSpec::dense(dir)).unwrap_or_else(|e| panic!("load {dir}: {e}"));
    assert!(
        provider.is_quantized(),
        "{dir}: the persisted q4 `quantization` block must load quantized projections"
    );
    let model = provider
        .causal_lm()
        .expect("YuE checkpoints are LlamaForCausalLM");
    let vocab = usize::try_from(model.config().vocab_size).expect("positive vocab_size");
    let ids = Tensor::from_vec(prompt.to_vec(), (1, prompt.len()), model.device()).unwrap();
    let mut cache = model.new_cache();
    let logits = model
        .decode_logits(&ids, &mut cache, 0)
        .unwrap_or_else(|e| panic!("forward {dir}: {e}"));
    assert_eq!(
        logits.dims(),
        &[1, vocab],
        "{dir}: last-position logits shape"
    );
    let row = logits
        .to_dtype(DType::F32)
        .unwrap()
        .to_device(&Device::Cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    (provider, row, vocab)
}

fn assert_finite_and_informative(tag: &str, row: &[f32]) {
    let non_finite = row.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(non_finite, 0, "{tag}: {non_finite} non-finite logits");
    let (min, max) = row
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        });
    assert!(max > min, "{tag}: logits are constant ({min})");
    let argmax = row
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap();
    eprintln!(
        "{tag}: width {} min {min:.3} max {max:.3} argmax {argmax}",
        row.len()
    );
}

#[test]
#[ignore = "needs a prepared YuE q4 stage-1 snapshot via YUE_S1_Q4_SNAPSHOT"]
fn yue_stage1_q4_prepared_snapshot_loads_and_yields_finite_logits() {
    let dir = snapshot("YUE_S1_Q4_SNAPSHOT");
    let (_provider, row, vocab) = last_logits(&dir, STAGE1_PROMPT);
    assert_eq!(vocab, 83_968, "stage-1 vocab width");
    assert_finite_and_informative("stage-1 q4", &row);
}

#[test]
#[ignore = "needs a prepared YuE q4 stage-2 snapshot via YUE_S2_Q4_SNAPSHOT"]
fn yue_stage2_q4_prepared_snapshot_loads_and_yields_finite_logits() {
    let dir = snapshot("YUE_S2_Q4_SNAPSHOT");
    let (_provider, row, vocab) = last_logits(&dir, STAGE2_PROMPT);
    assert_eq!(vocab, 83_840, "stage-2 vocab width");
    assert_finite_and_informative("stage-2 q4", &row);
}
