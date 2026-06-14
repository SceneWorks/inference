//! Real-weights prompt-refine smoke driver.
//!
//! Loads a Llama-3.2-3B-Instruct snapshot and rewrites a prompt the way the worker's `prompt_refine`
//! job will, printing the input + refined text. Run on Apple Silicon with a real snapshot:
//!   PROMPT_REFINE_MODEL_DIR=/path/to/snapshot \
//!     cargo run -p mlx-gen-prompt-refine --example refine -- "a cat sitting on a windowsill"

use mlx_gen::gen_core::{TextLlmRequest, TextLlmSampling};
use mlx_gen::{LoadSpec, Progress, WeightsSource};

/// A representative prompt-rewrite system message (the worker folds its real rewrite rules + the
/// active model's prompt guide into `system`; this is just enough to exercise the path end-to-end).
const SYSTEM: &str = "You are a prompt engineer for a text-to-image model. Rewrite the user's prompt \
into a single, vivid, comma-light sentence describing the subject, its key attributes, the setting, \
and the lighting. Preserve the user's intent. Output only the rewritten prompt, with no preamble, \
quotes, or explanation.";

fn main() {
    let dir = std::env::var("PROMPT_REFINE_MODEL_DIR")
        .expect("set PROMPT_REFINE_MODEL_DIR to a Llama-3.2-3B-Instruct snapshot directory");
    let user = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "a cat sitting on a windowsill".to_owned());

    let spec = LoadSpec::new(WeightsSource::Dir(dir.into()));
    let refiner = mlx_gen_prompt_refine::load(&spec).expect("load prompt_refine provider");

    let req = TextLlmRequest {
        system: SYSTEM.to_owned(),
        prompt: user.clone(),
        sampling: TextLlmSampling {
            seed: Some(42),
            ..Default::default()
        },
        ..Default::default()
    };

    let mut steps = 0u32;
    let out = refiner
        .generate(&req, &mut |p| {
            if let Progress::Step { current, .. } = p {
                steps = current;
            }
        })
        .expect("generate");

    println!("input:   {user}");
    println!("refined: {}", out.text);
    println!(
        "({} tokens, {steps} progress steps, finish {:?})",
        out.generated_tokens.unwrap_or(0),
        out.finish_reason
    );
}
