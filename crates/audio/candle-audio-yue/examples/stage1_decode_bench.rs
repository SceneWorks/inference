//! Stage-1 decode-cost micro-benchmark (not a test gate): loads a staged stage-1 snapshot, feeds
//! one synthetic segment prompt and decodes `TOKENS` tokens under CFG (batch of 2, the production
//! path), printing the wall time of every 100 steps. A per-step cost that grows with the position
//! is the O(n²) signature; a flat one is the O(1)-per-step (beyond attention) contract.
//!
//! ```text
//! cargo run --release -p candle-audio-yue --example stage1_decode_bench -- \
//!   ~/.cache/sceneworks-yue-assets/yue-s1-7b-anneal-en-cot-candle q4 1000 [prompt_len]
//! ```
//!
//! Run real weights under an external RSS watchdog. `<EOA>` is barred for the whole budget
//! (`min_new_tokens = max_new_tokens`) so every run decodes exactly `TOKENS` steps.

use std::path::PathBuf;
use std::time::Instant;

use candle_audio_yue::config::DecodeConfig;
use candle_audio_yue::stage1::{SegmentStart, Stage1Lm, Stage1Model};
use candle_audio_yue::Tier;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let root = PathBuf::from(
        args.first()
            .expect("usage: <stage-1 root> <q4|q8|bf16> <tokens>"),
    );
    let tier = match args.get(1).map(String::as_str) {
        Some("q4") | None => Some(Tier::Q4),
        Some("q8") => Some(Tier::Q8),
        Some("bf16") => None,
        Some(other) => panic!("tier must be q4, q8 or bf16, got {other}"),
    };
    let tokens: u32 = args.get(2).map_or(1000, |s| s.parse().expect("tokens"));
    let prompt_len: usize = args.get(3).map_or(400, |s| s.parse().expect("prompt_len"));

    let t = Instant::now();
    let mut lm = Stage1Lm::load(&root, tier).expect("load stage 1");
    eprintln!("load {:.1}s", t.elapsed().as_secs_f64());

    // A deterministic synthetic prompt in the text-token range; the decode cost does not depend on
    // what the tokens are, only on how many positions the cache holds.
    let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
    let prompt: Vec<u32> = (0..prompt_len)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            100 + (x % 30_000) as u32
        })
        .collect();
    let decode = DecodeConfig {
        max_new_tokens: tokens,
        min_new_tokens: tokens,
        ..DecodeConfig::default()
    };
    let bound = candle_audio_yue::stage1::render_positions([prompt.as_slice()], tokens);
    lm.begin_render(42, bound).expect("begin render");
    let t = Instant::now();
    lm.begin_segment(&SegmentStart {
        index: 0,
        prompt: &prompt,
        guidance_scale: Some(1.5),
        decode: &decode,
    })
    .expect("begin segment");
    eprintln!(
        "prefill {prompt_len} tokens {:.2}s",
        t.elapsed().as_secs_f64()
    );

    let start = Instant::now();
    let mut window = Instant::now();
    for i in 1..=tokens {
        lm.step().expect("step");
        if i % 100 == 0 {
            eprintln!(
                "steps {:>5}  last-100 {:>7.2}s  ({:>6.1} ms/step)  total {:>8.1}s",
                i,
                window.elapsed().as_secs_f64(),
                window.elapsed().as_secs_f64() * 10.0,
                start.elapsed().as_secs_f64()
            );
            window = Instant::now();
        }
    }
    lm.end_segment().expect("end segment");
    eprintln!(
        "decoded {tokens} tokens in {:.1}s",
        start.elapsed().as_secs_f64()
    );
}
