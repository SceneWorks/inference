//! DiT parity vs the frozen `QwenImage21Transformer2DModel` on the committed miniature snapshot
//! (`mlx-gen-qwen-image-2-1/tools/dump_qwen21_transformer.py`): the joint RoPE table, every named
//! stage (`txt_in`, `img_in`, `temb`, `modulation`, each block's attention / MLP / output,
//! `norm_out`, `proj_out`) and the target velocity, for a square target, a non-square two-slot
//! target, and the `t = 0` row. The candle twin of `mlx-gen-qwen-image-2-1`'s
//! `tests/transformer_parity.rs`, reading the SAME committed fixture.
//!
//! Runs on the plain CPU backend in f32, so it is **not** `#[ignore]`d — no GPU, no real weights.
//!
//! Tolerances: **1e-5 absolute** on the RoPE table (host trig on both sides); **1e-2 × peak** on
//! every matmul-bearing stage and the velocity — the repository's stated bound for an f32 matmul
//! chain against f32 CPU torch, kept identical to the MLX twin's so the two gates are comparable.
//! candle CPU f32 vs torch CPU f32 is far tighter than that in practice: measured on this fixture,
//! the RoPE table is **bit-identical** (`max|Δ| = 0`) and every matmul-bearing stage of every case
//! lands at **≤ 1.5e-5 × peak** (worst: `square/block_1_mlp`, `max|Δ| = 7.0e-5` at `peak = 4.8`;
//! the velocity worst case is `square`, `4.9e-5` at `peak = 3.3` → `1.5e-5 × peak`). `assert_close`
//! prints the `max|Δ|` / `mean|Δ|` / `peak` of every stage — run with `--nocapture`. A stage past
//! ~1e-4 × peak would be a porting bug, not tolerance, and the trace localises it.

use candle_core::{Device, IndexOp};
use candle_gen_qwen_image_2_1::loader::load_transformer;
use candle_gen_qwen_image_2_1::JointLayout;

use crate::common::{assert_close, host_f32, tiny_snapshot, Fixture};

const TOL: f32 = 1e-2;

#[test]
fn rope_every_stage_and_velocity_match_upstream() {
    let w = Fixture::open("qwen21_transformer.safetensors");
    let model = load_transformer(&tiny_snapshot(), &Device::Cpu).unwrap();
    for case in ["square", "wide", "square_t0"] {
        let height = w.meta_usize(&format!("{case}/height"));
        let width = w.meta_usize(&format!("{case}/width"));
        let text_len = w.meta_usize(&format!("{case}/text_len"));
        let timestep = w.meta_f32(&format!("{case}/timestep"));
        let hidden = w.tensor(&format!("{case}/hidden_states"));
        let text = w.tensor(&format!("{case}/encoder_hidden_states"));
        assert_eq!(text.dim(1).unwrap(), text_len);
        let layout = JointLayout::text_to_image(text_len, height, width);

        let (cos, sin) = model.rope(&layout).unwrap();
        let (got_cos, got_sin) = (host_f32(&cos), host_f32(&sin));
        let want_cos = host_f32(w.tensor(&format!("{case}/rope_cos")));
        let want_sin = host_f32(w.tensor(&format!("{case}/rope_sin")));
        assert_eq!(got_cos.len(), want_cos.len(), "{case}: rope width");
        let mut rope_max = 0f32;
        for (i, ((gc, wc), (gs, ws))) in got_cos
            .iter()
            .zip(&want_cos)
            .zip(got_sin.iter().zip(&want_sin))
            .enumerate()
        {
            rope_max = rope_max.max((gc - wc).abs()).max((gs - ws).abs());
            assert!(
                (gc - wc).abs() <= 1e-5 && (gs - ws).abs() <= 1e-5,
                "{case}: rope[{i}] ({gc}, {gs}) vs ({wc}, {ws})"
            );
        }
        eprintln!("{case}/rope: max|Δ|={rope_max:.3e} bound=1.000e-5");

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
            let want = w.tensor(&key);
            // Joint-sequence stages are `[1, S, C]` here and `[S, C]` in the fixture; the timestep
            // stages are `[rows, C]` on both sides.
            let got = if got.dims().len() == 3 {
                got.i(0).unwrap()
            } else {
                got.clone()
            };
            assert_close(&format!("{case}/{stage}"), &got, want, TOL);
        }

        // The model emits the whole joint sequence; the pipeline keeps the target's tail.
        let want = w.tensor(&format!("{case}/velocity"));
        let total = want.dim(0).unwrap();
        let target = layout.target_tokens();
        assert_close(
            &format!("{case}/velocity"),
            &velocity.i(0).unwrap(),
            &want.i(total - target..total).unwrap(),
            TOL,
        );
    }
}
