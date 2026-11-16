//! Decode-path throughput bench for a resident Qwen3.5-family (`qwen3_5`) snapshot.
//!
//! Loads the snapshot once and decodes one text prompt per case: greedy, the describer's
//! temperature (0.4, top-p 0.9), the JSON constraint, and each of those with an MTP proposer.
//! Prints tokens/s, the sampler and speculative acceptance per case, plus the greedy token ids'
//! hash so a change that keeps the math can be checked for token identity.
//!
//! ```text
//! CANDLE_LLM_QWEN35_MODEL=<snapshot dir> \
//!   cargo run --release --features cuda --example qwen35_decode_bench -- [cases] [max_new_tokens]
//! ```
//!
//! `cases` is a `,`- or `+`-separated list from `greedy,t04,json,greedy_mtp,t04_mtp,json_mtp` (default: all).

use core_llm::{
    Constraint, LoadSpec, Message, Sampling, Speculative, SpeculativeProposer, StreamEvent,
    TextLlmRequest, ThinkingMode,
};

const PROMPT: &str = "Describe, as a JSON object with the keys \"subject\", \"setting\", \
    \"lighting\", \"mood\" and \"details\", an old lighthouse on a rocky coast at dusk during a \
    storm. Each value is a long paragraph of at least eighty words.";

fn main() {
    let dir = std::env::var("CANDLE_LLM_QWEN35_MODEL").expect("set CANDLE_LLM_QWEN35_MODEL");
    let args: Vec<String> = std::env::args().collect();
    let cases = args
        .get(1)
        .cloned()
        .unwrap_or_else(|| "greedy,t04,json,greedy_mtp,t04_mtp,json_mtp".into());
    let max_new: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(400);
    let mtp_depth: u32 = std::env::var("BENCH_MTP_DEPTH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);

    let t = std::time::Instant::now();
    let model = candle_llm::load_for_model(&LoadSpec {
        source: dir.clone(),
        ..Default::default()
    })
    .expect("load");
    eprintln!("LOAD {:.1}s", t.elapsed().as_secs_f64());

    let mtp = Some(Speculative::proposer(SpeculativeProposer::Mtp, mtp_depth));
    // Warm-up (graph capture, kernel JIT).
    run(&*model, "warmup", 0.0, None, None, 64);
    for case in cases.split([',', '+']) {
        let (temp, constraint, spec) = match case {
            "greedy" => (0.0, None, None),
            "t04" => (0.4, None, None),
            "json" => (0.4, Some(Constraint::Json), None),
            "greedy_json" => (0.0, Some(Constraint::Json), None),
            "greedy_mtp" => (0.0, None, mtp),
            "t04_mtp" => (0.4, None, mtp),
            "json_mtp" => (0.4, Some(Constraint::Json), mtp),
            "greedy_json_mtp" => (0.0, Some(Constraint::Json), mtp),
            other => panic!("unknown case {other}"),
        };
        run(&*model, case, temp, constraint, spec, max_new);
    }
}

fn run(
    model: &dyn core_llm::TextLlm,
    label: &str,
    temperature: f32,
    constraint: Option<Constraint>,
    speculative: Option<Speculative>,
    max_new: u32,
) {
    let request = TextLlmRequest {
        messages: vec![Message::user(PROMPT.to_owned())],
        sampling: Sampling {
            temperature,
            top_p: if temperature > 0.0 { 0.9 } else { 1.0 },
            ..Sampling::default()
        },
        max_new_tokens: max_new,
        constraint,
        seed: Some(7),
        thinking: ThinkingMode::Disabled,
        speculative,
        ..Default::default()
    };
    let mut text = String::new();
    let out = model
        .generate(&request, &mut |ev| {
            if let StreamEvent::Token { text: t, .. } = ev {
                text.push_str(&t);
            }
        })
        .expect("generate");
    let decode = out.timings.as_ref().map(|t| t.decode.as_secs_f64()).unwrap_or(0.0);
    let prefill = out.timings.as_ref().map(|t| t.prefill.as_secs_f64()).unwrap_or(0.0);
    let r = out.decode.as_ref();
    // FNV-1a of the decoded text: equal hashes = identical token stream (for parity checks).
    let mut h: u64 = 0xcbf29ce484222325;
    for b in text.bytes() {
        h = (h ^ u64::from(b)).wrapping_mul(0x100000001b3);
    }
    eprintln!(
        "DECODE case={label} gen={} prefill_s={prefill:.3} decode_s={decode:.2} tok_s={:.2} \
         sampler={:?} graphs={:?}/{:?} proposer={:?} proposed={:?} accepted={:?} fwd={:?} \
         verify={:?} demoted={:?} text_hash={h:016x}",
        out.usage.generated_tokens,
        f64::from(out.usage.generated_tokens) / decode.max(1e-9),
        r.map(|r| r.sampler.clone()),
        r.map(|r| r.cuda_graphs.replayed),
        r.map(|r| r.cuda_graphs.eager),
        r.map(|r| r.proposer),
        r.map(|r| r.proposed_tokens),
        r.map(|r| r.accepted_tokens),
        r.map(|r| r.target_forwards),
        r.map(|r| r.verify_steps),
        r.and_then(|r| r.speculative_demoted_at),
    );
    if std::env::var("BENCH_PRINT").is_ok() {
        eprintln!("TEXT[{label}] {text}");
    }
}
