//! sc-5109 acceptance gate: the Lens harmony render + tokenizer must produce byte-identical ids to
//! the HF reference (`apply_chat_template` over the [system, user, assistant-thinking] conversation).
//!
//! Goldens are dumped by `scripts/dump_tokenizer_goldens.py` (lens-venv). Light + CPU-only, but it
//! needs the model's `tokenizer.json` + the goldens, so it is gated on env vars and skips cleanly:
//!   LENS_TOKENIZER_JSON — the Lens `tokenizer/tokenizer.json`
//!   LENS_TOK_GOLDENS    — tokenizer_goldens.safetensors
//! Run: cargo test -p candle-gen-lens --test tokenizer_parity -- --nocapture

use candle_gen::candle_core::{Device, Tensor};
use candle_gen_lens::text::{LensTokenizer, TXT_OFFSET};

type R = Result<(), Box<dyn std::error::Error>>;

/// Must match `scripts/dump_tokenizer_goldens.py::PROMPTS` exactly (the golden ids are keyed by index).
const PROMPTS: &[&str] = &[
    "a red cube on a wooden table",
    "X",
    "猫が窓辺で眠っている",
    "A photorealistic wide-angle photograph of a bustling Tokyo street at night in the rain, \
     neon signs reflecting off the wet asphalt, dozens of pedestrians with transparent umbrellas, \
     a yellow taxi at a crosswalk, steam rising from a ramen stall, cinematic shallow depth of field.",
];

fn ids_u32(t: &Tensor) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    Ok(t.to_vec1::<i64>()?.into_iter().map(|i| i as u32).collect())
}

#[test]
fn lens_tokenizer_matches_hf_reference() -> R {
    let Ok(tok_json) = std::env::var("LENS_TOKENIZER_JSON") else {
        eprintln!("SKIP: set LENS_TOKENIZER_JSON to the Lens tokenizer/tokenizer.json");
        return Ok(());
    };
    let Ok(goldens_path) = std::env::var("LENS_TOK_GOLDENS") else {
        eprintln!("SKIP: set LENS_TOK_GOLDENS to tokenizer_goldens.safetensors");
        return Ok(());
    };
    if !std::path::Path::new(&goldens_path).exists() {
        eprintln!("SKIP: goldens not found at {goldens_path}");
        return Ok(());
    }

    let goldens = candle_gen::candle_core::safetensors::load(&goldens_path, &Device::Cpu)?;
    let date = String::from_utf8(goldens["date_utf8"].to_vec1::<u8>()?)?;
    eprintln!("date: {date}");

    let tok = LensTokenizer::from_file(&tok_json)?;

    let mut encoded: Vec<Vec<u32>> = Vec::new();
    for (i, prompt) in PROMPTS.iter().enumerate() {
        let mine = tok.encode(prompt, &date)?;
        let golden = ids_u32(&goldens[&format!("ids_{i}")])?;
        assert_eq!(
            mine,
            golden,
            "prompt {i} ids diverge (len mine={} golden={})",
            mine.len(),
            golden.len()
        );
        eprintln!("prompt {i}: {} ids match", mine.len());
        encoded.push(mine);
    }

    // txt_offset = 97: the harmony preamble is prompt-independent, so the first TXT_OFFSET ids are
    // identical across prompts and the user content begins exactly at TXT_OFFSET.
    assert!(encoded[0].len() > TXT_OFFSET && encoded[1].len() > TXT_OFFSET);
    assert_eq!(
        encoded[0][..TXT_OFFSET],
        encoded[1][..TXT_OFFSET],
        "the {TXT_OFFSET}-token preamble must be prompt-independent"
    );
    assert_ne!(
        encoded[0][TXT_OFFSET], encoded[1][TXT_OFFSET],
        "user content must begin at txt_offset {TXT_OFFSET} (differs by prompt)"
    );
    eprintln!(
        "PASS — {} prompts byte-identical; txt_offset {TXT_OFFSET} confirmed",
        PROMPTS.len()
    );
    Ok(())
}
