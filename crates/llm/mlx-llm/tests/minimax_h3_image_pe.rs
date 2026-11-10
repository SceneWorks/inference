//! MiniMax-H3-Image-PE real-weight evidence (SW-7099, feature SW-F363): AC1 (the expander runs
//! t2i and edit-with-references on this backend) and AC2 (each converted tier states its quality
//! against the NVFP4 original on a fixed prompt set).
//!
//! One `#[ignore]`d test runs `core_llm::minimax_h3_image_pe::quality`'s fixed set greedily on
//! the tier at `MINIMAX_PE_TIER` and writes one JSON document. Labelled `nvfp4`, the run IS the
//! reference (the published ModelOpt weights, natively on a Blackwell card); any other label is
//! scored against the reference document by caption chrF, aspect-ratio agreement and parse rate.
//! The test fails when any case does not produce a schema-valid expansion; the quality numbers
//! are stated, not gated.
//!
//! | variable | meaning |
//! |---|---|
//! | `MINIMAX_PE_PROMPTS` | the pinned `MiniMaxAI/MiniMax-H3-Image-PE` snapshot (holds `prompt/`) |
//! | `MINIMAX_PE_TIER` | the snapshot to load (the NVFP4 source itself, or a converted tier) |
//! | `MINIMAX_PE_LABEL` | `nvfp4`, or the tier's label (`candle-bf16`, `mlx-q4`, …) |
//! | `MINIMAX_PE_REFERENCE` | the `nvfp4` run's document (every label but `nvfp4`) |
//! | `MINIMAX_PE_OUTPUT` | JSON path to write (must not exist) |

use std::path::{Path, PathBuf};

use core_llm::minimax_h3_image_pe::{quality, PromptFiles};
use core_llm::LoadSpec;
use mlx_llm::LlamaProvider;

fn var(name: &str) -> String {
    std::env::var(name)
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| panic!("{name} must be set"))
}

#[test]
#[ignore = "needs MiniMax-H3-Image-PE weights: see the module docs for the MINIMAX_PE_* variables"]
fn minimax_h3_image_pe_quality_against_nvfp4() {
    let files = PromptFiles::load_verified(Path::new(&var("MINIMAX_PE_PROMPTS")))
        .expect("verified MiniMax-H3-Image-PE prompts");
    let label = var("MINIMAX_PE_LABEL");
    let output = PathBuf::from(var("MINIMAX_PE_OUTPUT"));
    assert!(!output.exists(), "{} already exists", output.display());
    let provider =
        LlamaProvider::load(&LoadSpec::dense(var("MINIMAX_PE_TIER"))).expect("load tier");
    let runs = quality::run_set(&provider, &files).expect("run the fixed set");
    let mut doc = quality::runs_to_json(&label, &runs);
    if label != "nvfp4" {
        let reference: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(var("MINIMAX_PE_REFERENCE")).expect("read reference"),
        )
        .expect("reference JSON");
        let reference = quality::reference_from_json(&reference).expect("reference cases");
        let (scores, summary) = quality::compare(&runs, &reference).expect("same case set");
        doc["versus_nvfp4"] = serde_json::json!({
            "metric": "caption chrF (char 6-gram, beta 2), aspect-ratio agreement, parse rate",
            "cases": scores.iter().map(|(id, s)| serde_json::json!({
                "id": id, "parsed": s.parsed, "ratio_matches": s.ratio_matches, "chrf": s.chrf,
            })).collect::<Vec<_>>(),
            "parse_rate": summary.parse_rate,
            "ratio_agreement": summary.ratio_agreement,
            "mean_chrf": summary.mean_chrf,
        });
    }
    std::fs::write(&output, serde_json::to_string_pretty(&doc).unwrap()).expect("write evidence");
    println!("{}", serde_json::to_string_pretty(&doc).unwrap());
    let failed: Vec<_> = runs
        .iter()
        .filter(|r| r.expansion.is_none())
        .map(|r| (r.id.as_str(), r.error.as_deref().unwrap_or("")))
        .collect();
    assert!(
        failed.is_empty(),
        "{label}: replies that broke the schema: {failed:?}"
    );
}
