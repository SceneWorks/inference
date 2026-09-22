//! DiT parity vs the frozen `QwenImage21Transformer2DModel` on the committed miniature snapshot
//! (`tools/dump_qwen21_transformer.py`): the joint RoPE table, every named stage (`img_in`,
//! `txt_in`, `temb`, `modulation`, each block's attention / MLP / output, `norm_out`, `proj_out`)
//! and the target velocity, for a square target, a non-square two-slot target, and the `t = 0`
//! row.
//!
//! Tolerances: **1e-5 absolute** on the RoPE table (host trig on both sides); **1e-2 × peak** on
//! every matmul-bearing stage and the velocity — the repository's stated bound for f32 Metal
//! matmul chains vs f32 CPU torch (MLX runs f32 matmul in reduced precision, ~1e-3 per op, and a
//! block compounds several). Measured on this fixture: ≤ 5e-3 × peak after two blocks; every stage
//! prints its own numbers.

use mlx_gen_qwen_image_2_1::{load_transformer, JointLayout};
use mlx_rs::ops::indexing::IndexOp;

use crate::common::{assert_close, fixture, host_f32, meta_f32, meta_usize, tiny_snapshot};

const TOL: f32 = 1e-2;

#[test]
fn rope_every_stage_and_velocity_match_upstream() {
    let w = fixture("qwen21_transformer.safetensors");
    let model = load_transformer(&tiny_snapshot()).unwrap();
    for case in ["square", "wide", "square_t0"] {
        let height = meta_usize(&w, &format!("{case}/height"));
        let width = meta_usize(&w, &format!("{case}/width"));
        let text_len = meta_usize(&w, &format!("{case}/text_len"));
        let timestep = meta_f32(&w, &format!("{case}/timestep"));
        let hidden = w.require(&format!("{case}/hidden_states")).unwrap();
        let text = w.require(&format!("{case}/encoder_hidden_states")).unwrap();
        assert_eq!(text.shape()[1] as usize, text_len);
        let layout = JointLayout::text_to_image(text_len, height, width);

        let (cos, sin) = model.rope(&layout).unwrap();
        let (got_cos, got_sin) = (host_f32(&cos), host_f32(&sin));
        let want_cos = host_f32(w.require(&format!("{case}/rope_cos")).unwrap());
        let want_sin = host_f32(w.require(&format!("{case}/rope_sin")).unwrap());
        assert_eq!(got_cos.len(), want_cos.len(), "{case}: rope width");
        for (i, ((gc, wc), (gs, ws))) in got_cos
            .iter()
            .zip(&want_cos)
            .zip(got_sin.iter().zip(&want_sin))
            .enumerate()
        {
            assert!(
                (gc - wc).abs() <= 1e-5 && (gs - ws).abs() <= 1e-5,
                "{case}: rope[{i}] ({gc}, {gs}) vs ({wc}, {ws})"
            );
        }

        let (velocity, trace) = model
            .forward_joint_traced(text, &[hidden], timestep, &layout)
            .unwrap();
        assert_eq!(trace.len(), 6 + 3 * model.config().num_layers);
        for (stage, got) in &trace {
            // Block outputs come from the block hook (`block_{i}_out`); every other stage from
            // its module hook (`trace/<stage>`).
            let key = if stage.ends_with("_out") && stage.starts_with("block_") {
                format!("{case}/{stage}")
            } else {
                format!("{case}/trace/{stage}")
            };
            let want = w
                .require(&key)
                .unwrap_or_else(|_| panic!("{case}: fixture lacks {key}"));
            // Joint-sequence stages are `[1, S, C]` here and `[S, C]` in the fixture; the timestep
            // stages are `[rows, C]` on both sides.
            let got = if got.shape().len() == 3 {
                got.index((0, .., ..))
            } else {
                got.clone()
            };
            assert_close(&format!("{case}/{stage}"), &got, want, TOL);
        }

        // The model emits the whole joint sequence; the pipeline keeps the target's tail.
        let want = w.require(&format!("{case}/velocity")).unwrap();
        let total = want.shape()[0];
        let target = layout.target_tokens() as i32;
        assert_close(
            &format!("{case}/velocity"),
            &velocity.index((0, .., ..)),
            &want.index((total - target..total, ..)),
            TOL,
        );
    }
}
