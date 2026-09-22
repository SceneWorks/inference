//! Text-conditioning parity vs the frozen pipeline's `encode_prompt` on the committed miniature
//! snapshot (`tools/dump_qwen21_text_encoder.py`): template token ids, the derived system-prefix
//! drop count, the embedding and every decoder layer's output, the last-layer pre-norm hidden
//! state, and the final prompt embeddings — for a normal prompt, a negative prompt, the empty
//! prompt and an RGBA-style prompt.
//!
//! Tolerance: **1e-2 × peak** on every layer and the embeddings — measured ≤ **1.8e-3 × peak**
//! (`layer_1`, 1.44e-3 on a 0.84 peak: f32 CPU torch vs the reduced-precision f32 Metal matmul, two
//! GQA layers deep), 5.5× headroom; the embedding lookup itself is held to **1e-6** since it is a
//! gather (measured 0).

use mlx_gen_qwen_image_2_1::{
    load_text_encoder, load_tokenizer, prompt_template, system_prompt_drop_count,
};
use mlx_rs::ops::indexing::IndexOp;

use crate::common::{assert_close, fixture, host_f32, meta_str, meta_usize, tiny_snapshot};

const TOL: f32 = 1e-2;

#[test]
fn template_ids_drop_count_layers_and_embeddings_match_upstream() {
    let w = fixture("qwen21_text_encoder.safetensors");
    let root = tiny_snapshot();
    let tokenizer = load_tokenizer(&root).unwrap();
    let encoder = load_text_encoder(&root).unwrap();

    let drop = system_prompt_drop_count(&tokenizer).unwrap();
    assert_eq!(drop, meta_usize(&w, "drop_idx"), "system-prefix drop count");
    assert_eq!(
        meta_str(&w, "sys_prompt"),
        mlx_gen_qwen_image_2_1::SYSTEM_PROMPT
    );

    for case in ["fox", "negative", "empty", "rgba"] {
        let prompt = meta_str(&w, &format!("{case}/prompt"));
        let want_ids: Vec<i32> = host_f32(w.require(&format!("{case}/input_ids")).unwrap())
            .into_iter()
            .map(|v| v as i32)
            .collect();
        let tokens = tokenizer
            .tokenize_preformatted(&prompt_template(prompt))
            .unwrap();
        assert_eq!(tokens.ids, want_ids, "{case}: template token ids");

        let (ids, mask) = mlx_gen::tokenizer::to_arrays(&tokens);
        let (hidden, stages) = encoder.forward_traced(&ids, &mask, true).unwrap();
        for (stage, got) in &stages {
            let tol = if stage == "embed" { 1e-6 } else { TOL };
            assert_close(
                &format!("{case}/{stage}"),
                &got.index((0, .., ..)),
                w.require(&format!("{case}/trace/{stage}")).unwrap(),
                tol,
            );
        }
        assert_close(
            &format!("{case}/last_hidden"),
            &hidden.index((0, .., ..)),
            w.require(&format!("{case}/last_hidden")).unwrap(),
            TOL,
        );

        let embeds = encoder.encode_prompt(&tokenizer, prompt, drop).unwrap();
        let want = w.require(&format!("{case}/prompt_embeds")).unwrap();
        assert_eq!(embeds.shape()[1], want.shape()[0], "{case}: dropped length");
        assert_close(
            &format!("{case}/prompt_embeds"),
            &embeds.index((0, .., ..)),
            want,
            TOL,
        );
    }
}
