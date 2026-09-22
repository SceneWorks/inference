//! Text-conditioning parity vs the frozen pipeline's `encode_prompt` on the committed miniature
//! snapshot (`crates/media/mlx-gen/tools/dump_qwen21_text_encoder.py`): template token ids, the
//! derived system-prefix drop count, the embedding and every decoder layer's output, the last-layer
//! pre-norm hidden state, and the final prompt embeddings — for a normal prompt, a negative prompt,
//! the empty prompt and an RGBA-style prompt.
//!
//! Fixture and snapshot are the SAME ones the MLX twin reads, so the two backends' conditioning is
//! pinned to one reference.
//!
//! Tolerance: **1e-5 × peak** on every layer and the embeddings — candle CPU f32 against torch CPU
//! f32 is the same arithmetic on the same hardware, so the MLX twin's 1e-2 (which budgets Metal's
//! reduced-precision f32 matmul) is far looser than this lane needs. Measured on this fixture:
//! ≤ 2.3e-7 absolute at every stage (peak ≈ 0.84), i.e. ~2.7e-7 × peak — a ~40× margin. The
//! embedding lookup itself is held to **1e-6** and measures exactly 0: it is a gather. Every stage
//! prints its measured numbers.

use candle_core::{DType, IndexOp};
use candle_gen_qwen_image_2_1::{
    load_text_encoder, load_tokenizer, prompt_template, system_prompt_drop_count,
};

use crate::common::{assert_close, device, host_f32, tiny_snapshot, Fixture};

const TOL: f32 = 1e-5;

#[test]
fn template_ids_drop_count_layers_and_embeddings_match_upstream() {
    let w = Fixture::open("qwen21_text_encoder.safetensors");
    let root = tiny_snapshot();
    let dev = device();
    let tokenizer = load_tokenizer(&root).unwrap();
    let encoder = load_text_encoder(&root, &dev).unwrap();

    let drop = system_prompt_drop_count(&tokenizer).unwrap();
    assert_eq!(drop, w.meta_usize("drop_idx"), "system-prefix drop count");
    assert_eq!(
        w.meta("sys_prompt"),
        candle_gen_qwen_image_2_1::SYSTEM_PROMPT
    );

    for case in ["fox", "negative", "empty", "rgba"] {
        let prompt = w.meta(&format!("{case}/prompt")).to_owned();
        let want_ids: Vec<i32> = host_f32(&w.tensor(&format!("{case}/input_ids")))
            .into_iter()
            .map(|v| v as i32)
            .collect();
        let tokens = tokenizer
            .tokenize_preformatted(&prompt_template(&prompt))
            .unwrap();
        assert_eq!(tokens.ids, want_ids, "{case}: template token ids");

        let ids = candle_gen_qwen_image_2_1::text_encoder::input_ids(&tokens.ids, &dev).unwrap();
        let (hidden, stages) = encoder.forward_traced(&ids, true).unwrap();
        for (stage, got) in &stages {
            let tol = if stage == "embed" { 1e-6 } else { TOL };
            assert_close(
                &format!("{case}/{stage}"),
                &got.i(0).unwrap().to_dtype(DType::F32).unwrap(),
                &w.tensor(&format!("{case}/trace/{stage}")),
                tol,
            );
        }
        assert_close(
            &format!("{case}/last_hidden"),
            &hidden.i(0).unwrap().to_dtype(DType::F32).unwrap(),
            &w.tensor(&format!("{case}/last_hidden")),
            TOL,
        );

        let embeds = encoder.encode_prompt(&tokenizer, &prompt, drop).unwrap();
        let want = w.tensor(&format!("{case}/prompt_embeds"));
        assert_eq!(embeds.dims()[1], want.dims()[0], "{case}: dropped length");
        assert_close(
            &format!("{case}/prompt_embeds"),
            &embeds.i(0).unwrap().to_dtype(DType::F32).unwrap(),
            &want,
            TOL,
        );
    }
}
