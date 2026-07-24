//! sc-14040: whole-model NR-MMDiT parity against the frozen torch reference at **tiny dims in
//! f32** — committed fixture, no weights, no gitignored golden, runs in the default `cargo test`.
//!
//! ## Why this exists next to the real-weights gate
//!
//! `tests/dit_real_weights.rs` is the DoD oracle: real bf16 weights, the reference's own step-0
//! tensors. But the published checkpoint's block-0 modulation gates reach ~1e8, so twelve bf16
//! blocks amplify rounding to a **2e-2 mean-relative floor** — measured, and confirmed by the
//! port's own f32-vs-bf16 spread of 2.8e-2. Several real porting mistakes are *smaller* than that
//! floor at the output (substituting the `mlx-gen-z-image` sibling's SwiGLU gate for
//! `gelu-approximate` moves `dit_out` by only ~1.7e-2), so the real-weights gate cannot
//! discriminate them however the tolerance is drawn.
//!
//! This fixture removes the floor rather than arguing about it. `tools/dump_mage_flow_small.py`
//! runs the **vendored reference itself** — a 2-block `MageFlow` at dim 24 with seeded random
//! weights, in f32 — and dumps its state dict, inputs and outputs. In f32 the port agrees with the
//! reference to ~1e-6, so every mutation below misses by four to six orders of magnitude.
//!
//! Two packings are covered because they exercise different code:
//!
//! * `gen` — the fused-CFG generation pack: two attention segments, one `img_shapes` entry each.
//! * `edit` — the edit pack (`pipeline.py:517-519`): ONE attention segment carrying TWO
//!   `img_shapes` entries, which is the only configuration where the msrope **frame axis** changes
//!   the attention scores instead of cancelling out.

use mlx_rs::ops::concatenate_axis;
use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen_mage::config::MageFlowConfig;
use mlx_gen_mage::feed_forward::FfnActivation;
use mlx_gen_mage::{
    DualStream, ImgShape, MageTransformer, MsRope, PackContext, PackLayout, RopeTable,
};

mod common;
use common::error;

const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/mage_flow_small.safetensors"
);

/// f32 on Metal vs f32 on CPU. **Not** an equality: MLX runs f32 matmul in reduced precision on
/// Metal, so matmul chains agree to three or four significant figures — the same reason
/// `mlx-gen-z-image`'s block-parity fixture gates at 1e-2.
///
/// **Measured** on this fixture: whole 2-block model `max_rel` 2.4e-3 / `mean_rel` 2.6e-3 (`gen`)
/// and 1.7e-3 / 1.4e-3 (`edit`); one block 3.0e-3 / 2.8e-3; a single `img_in` Linear already sits
/// at 6.5e-4. The gate is ~4× the measurement — and, crucially, **two orders of magnitude tighter
/// than the bf16 real-weights floor**, which is the whole reason this fixture exists.
const F32_MAX_REL: f32 = 1.0e-2;
const F32_MEAN_REL: f32 = 1.0e-2;

/// To count as discriminated a mutation must (a) fail [`F32_MAX_REL`] outright and (b) sit at
/// least this multiple above the *unmutated* run's own error, so "caught" means caught above the
/// noise rather than caught by an unlucky tolerance.
const DISCRIMINATION_FACTOR: f32 = 5.0;

/// The msrope table is pure trig — no matmul, so no Metal reduced precision. **Measured** 6.0e-8
/// against the reference's own complex table, which is why it, not the model output, is where the
/// rotary conventions are gated.
const ROPE_TABLE_MAX_ABS: f32 = 1.0e-6;

fn fixture() -> Weights {
    Weights::from_file(FIXTURE).expect(
        "tests/fixtures/mage_flow_small.safetensors is committed; regenerate with \
         `<ref-venv>/bin/python crates/media/mlx-gen/tools/dump_mage_flow_small.py`",
    )
}

/// The fixture stores the state dict under a `model.` prefix so it can share a file with the
/// activations; strip it back to the checkpoint's own key layout.
fn model_weights(w: &Weights) -> Weights {
    let mut out = Weights::empty();
    for key in w.keys().collect::<Vec<_>>() {
        if let Some(rest) = key.strip_prefix("model.") {
            out.insert(rest, w.require(key).unwrap().clone());
        }
    }
    out
}

fn config(w: &Weights) -> MageFlowConfig {
    let c = w
        .require("config")
        .unwrap()
        .as_dtype(Dtype::Int32)
        .unwrap()
        .as_slice::<i32>()
        .to_vec();
    let cfg = MageFlowConfig {
        in_channels: c[0],
        out_channels: c[1],
        context_in_dim: c[2],
        hidden_size: c[3],
        num_heads: c[4],
        depth: c[5] as usize,
        patch_size: c[6],
        axes_dim: c[7..10].to_vec(),
        checkpoint: false,
    };
    cfg.validate().unwrap();
    cfg
}

fn ints(w: &Weights, key: &str) -> Vec<i32> {
    let t = w.require(key).unwrap_or_else(|e| panic!("{key}: {e}"));
    t.as_dtype(Dtype::Int32)
        .unwrap()
        .reshape(&[t.shape().iter().product::<i32>()])
        .unwrap()
        .as_slice::<i32>()
        .to_vec()
}

fn layout(w: &Weights, case: &str) -> PackLayout {
    let shapes: Vec<ImgShape> = ints(w, &format!("{case}.in.img_shapes"))
        .chunks(3)
        .map(|s| ImgShape::new(s[0], s[1], s[2]))
        .collect();
    let img_cu = ints(w, &format!("{case}.in.img_cu"));
    let txt_cu = ints(w, &format!("{case}.in.txt_cu"));
    let lens = |cu: &[i32]| cu.windows(2).map(|p| p[1] - p[0]).collect::<Vec<i32>>();
    let layout = PackLayout::new(shapes, lens(&img_cu), lens(&txt_cu)).unwrap();
    assert_eq!(layout.img_cu(), img_cu);
    assert_eq!(layout.txt_cu(), txt_cu);
    layout
}

/// Which msrope table a run should use — the mutation axis for the frame-index probe.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Rope {
    /// Built from the whole `img_shapes` list: entry `j` rotates at frame `j`.
    Reference,
    /// Every entry treated as if it were the first — the "frame index does not matter" reading.
    EveryShapeAtFrameZero,
    /// `scale_rope = false`: height/width start at 0 instead of being centred.
    UncentredSpatialAxes,
}

fn context(model: &MageTransformer, layout: PackLayout, rope: Rope) -> PackContext {
    match rope {
        Rope::Reference => model.pack_context(layout).unwrap(),
        Rope::UncentredSpatialAxes => {
            let cfg = model.config();
            let plain = MsRope::new(&cfg.axes_dim, 10_000.0, false, 4096).unwrap();
            PackContext::new(layout, &plain).unwrap()
        }
        Rope::EveryShapeAtFrameZero => {
            let rope = MsRope::from_config(model.config()).unwrap();
            let per: Vec<RopeTable> = layout
                .img_shapes()
                .iter()
                .map(|s| rope.forward(std::slice::from_ref(s)).unwrap())
                .collect();
            let cos: Vec<&Array> = per.iter().map(|t| &t.cos).collect();
            let sin: Vec<&Array> = per.iter().map(|t| &t.sin).collect();
            let table = RopeTable {
                cos: concatenate_axis(&cos, 0).unwrap(),
                sin: concatenate_axis(&sin, 0).unwrap(),
            };
            PackContext::with_rope_table(layout, table).unwrap()
        }
    }
}

fn run(case: &str, activation: FfnActivation, rope: Rope) -> (f32, f32, f32) {
    let w = fixture();
    let mut model = MageTransformer::from_weights(&model_weights(&w), config(&w)).unwrap();
    assert_eq!(model.dtype(), Dtype::Float32);
    model.set_ffn_activation(activation);
    let ctx = context(&model, layout(&w, case), rope);
    let out = model
        .forward(
            w.require(&format!("{case}.in.img")).unwrap(),
            w.require(&format!("{case}.in.txt")).unwrap(),
            w.require(&format!("{case}.in.timesteps")).unwrap(),
            &ctx,
        )
        .unwrap();
    error(&out, w.require(&format!("{case}.out")).unwrap())
}

/// The whole 2-block model, both packings, against the reference's own f32 output.
#[test]
fn small_model_matches_the_reference() {
    for case in ["gen", "edit"] {
        let (abs, rel, mean) = run(case, FfnActivation::GeluApproximate, Rope::Reference);
        println!("{case}: max_abs {abs:.4e} max_rel {rel:.4e} mean_rel {mean:.4e}");
        assert!(
            rel < F32_MAX_REL && mean < F32_MEAN_REL,
            "{case} diverged: max_rel {rel}, mean_rel {mean}"
        );
    }
}

/// One dual-stream block, fed the reference's own post-embedding streams, `temb` and msrope table.
#[test]
fn small_block_matches_the_reference() {
    let w = fixture();
    let model = MageTransformer::from_weights(&model_weights(&w), config(&w)).unwrap();
    for case in ["gen", "edit"] {
        let ctx = PackContext::with_rope_table(
            layout(&w, case),
            RopeTable {
                cos: w.require(&format!("{case}.rope_re")).unwrap().clone(),
                sin: w.require(&format!("{case}.rope_im")).unwrap().clone(),
            },
        )
        .unwrap();
        let stream = DualStream {
            img: w.require(&format!("{case}.block0_in.img")).unwrap().clone(),
            txt: w.require(&format!("{case}.block0_in.txt")).unwrap().clone(),
        };
        let temb = w.require(&format!("{case}.block0_in.temb")).unwrap();
        let out = model.blocks()[0].forward(&stream, temb, &ctx).unwrap();
        for (name, got, key) in [
            ("txt", &out.txt, format!("{case}.block0_out.txt")),
            ("img", &out.img, format!("{case}.block0_out.img")),
        ] {
            let (abs, rel, mean) = error(got, w.require(&key).unwrap());
            println!(
                "{case} block {name}: max_abs {abs:.4e} max_rel {rel:.4e} mean_rel {mean:.4e}"
            );
            assert!(
                rel < F32_MAX_REL && mean < F32_MEAN_REL,
                "{case} block {name} diverged: max_rel {rel}, mean_rel {mean}"
            );
        }
    }
}

/// The msrope table the port builds must equal the one the reference handed to its first block —
/// in both packings, including the edit pack where two shapes share one attention window.
///
/// **This is the sharp gate for every msrope convention.** The table is pure trig, so it agrees to
/// ~6e-8 and any wrong convention is off by O(1). That matters because the *output* is a blunt
/// instrument for absolute rotation: image↔image attention only sees relative positions, so a
/// convention that shifts every coordinate in a window by a constant — the fused-CFG frame index,
/// the `scale_rope` centring offset — cancels there and survives only through image↔text attention
/// (text is never rotated). Those leak into the output at ~1e-3, under the numerical floor. Here
/// they are caught outright.
#[test]
fn small_msrope_matches_the_reference_table() {
    let w = fixture();
    let model = MageTransformer::from_weights(&model_weights(&w), config(&w)).unwrap();
    for case in ["gen", "edit"] {
        let ctx = context(&model, layout(&w, case), Rope::Reference);
        for (name, got, key) in [
            ("cos", &ctx.rope().cos, format!("{case}.rope_re")),
            ("sin", &ctx.rope().sin, format!("{case}.rope_im")),
        ] {
            let (abs, ..) = error(got, w.require(&key).unwrap());
            println!("{case} msrope {name}: max_abs {abs:.4e}");
            assert!(
                abs < ROPE_TABLE_MAX_ABS,
                "{case} msrope {name} diverged: max_abs {abs}"
            );
        }
    }
}

/// ...and the table gate must reject both wrong conventions, by orders of magnitude.
#[test]
fn the_msrope_table_gate_rejects_both_wrong_conventions() {
    let w = fixture();
    let model = MageTransformer::from_weights(&model_weights(&w), config(&w)).unwrap();
    for (case, rope, what) in [
        (
            "gen",
            Rope::EveryShapeAtFrameZero,
            "every img_shapes entry rotated at frame 0",
        ),
        (
            "edit",
            Rope::EveryShapeAtFrameZero,
            "every img_shapes entry rotated at frame 0",
        ),
        (
            "gen",
            Rope::UncentredSpatialAxes,
            "un-centred spatial axes (scale_rope = false)",
        ),
    ] {
        let ctx = context(&model, layout(&w, case), rope);
        let (cos, ..) = error(
            &ctx.rope().cos,
            w.require(&format!("{case}.rope_re")).unwrap(),
        );
        let (sin, ..) = error(
            &ctx.rope().sin,
            w.require(&format!("{case}.rope_im")).unwrap(),
        );
        println!("{case}: {what} → table cos max_abs {cos:.4e} sin max_abs {sin:.4e}");
        assert!(
            cos.max(sin) > ROPE_TABLE_MAX_ABS * 1.0e5,
            "the msrope table gate does not reject {what:?} on the {case} pack \
             (cos {cos}, sin {sin})"
        );
    }
}

/// **The output-level discrimination battery.** Each row is a real porting mistake, with the
/// measured effect asserted *in both directions* — a mistake this gate cannot see is written down
/// as such rather than quietly omitted, so the suite never claims coverage it does not have.
#[test]
fn the_parity_gate_rejects_real_porting_mistakes() {
    struct Mutation {
        case: &'static str,
        what: &'static str,
        activation: FfnActivation,
        rope: Rope,
        /// `true` ⇒ must miss the gate by [`DISCRIMINATION_FACTOR`]; `false` ⇒ must stay *inside*
        /// it, because the difference is genuinely below the numerical floor at the output and the
        /// guard against it lives somewhere else (named in `note`).
        discriminated: bool,
        note: &'static str,
    }
    let mutations = [
        Mutation {
            case: "gen",
            what: "SwiGLU's SiLU gate instead of gelu-approximate",
            activation: FfnActivation::Silu,
            rope: Rope::Reference,
            discriminated: true,
            note: "the residue of inheriting the mlx-gen-z-image sibling's FFN",
        },
        Mutation {
            case: "edit",
            what: "every img_shapes entry rotated at msrope frame 0",
            activation: FfnActivation::GeluApproximate,
            rope: Rope::EveryShapeAtFrameZero,
            discriminated: true,
            note: "the frame axis is load-bearing exactly when one attention window spans several \
                   shapes, which is the edit stream [target, ref_1, ...]",
        },
        Mutation {
            case: "gen",
            what: "every img_shapes entry rotated at msrope frame 0",
            activation: FfnActivation::GeluApproximate,
            rope: Rope::EveryShapeAtFrameZero,
            discriminated: false,
            note: "on a one-shape-per-window pack the shift is a CONSTANT offset inside every \
                   window, so it cancels in image-image attention (RoPE is relative) and leaks \
                   only through image-text attention, text being unrotated. Guarded by \
                   the_msrope_table_gate_rejects_both_wrong_conventions. This is also why the \
                   sc-14036 inference that batch_cfg changes the render is wrong: the TABLE \
                   differs, the render does not (measured 1.1e-2 mean-relative on the real \
                   checkpoint, under its own 2.8e-2 bf16 sensitivity)",
        },
        Mutation {
            case: "gen",
            what: "un-centred msrope spatial axes (scale_rope = false)",
            activation: FfnActivation::GeluApproximate,
            rope: Rope::UncentredSpatialAxes,
            discriminated: false,
            note: "same reason: centring shifts every coordinate in a grid by the same constant. \
                   Guarded by the_msrope_table_gate_rejects_both_wrong_conventions",
        },
        Mutation {
            case: "gen",
            what: "exact/erf GELU instead of the tanh approximation",
            activation: FfnActivation::Gelu,
            rope: Rope::Reference,
            discriminated: false,
            note: "the two GELUs differ by <4e-4 absolute, i.e. below MLX's own f32 matmul floor \
                   — no parity gate at any scale separates them. The guard is structural: \
                   config::FFN_ACTIVATION and the names FfnActivation::from_name accepts, both \
                   pinned against the vendored reference in tests/config_conformance.rs",
        },
    ];
    for m in mutations {
        let (_, base, _) = run(m.case, FfnActivation::GeluApproximate, Rope::Reference);
        let (_, rel, mean) = run(m.case, m.activation, m.rope);
        println!(
            "{}: {} -> max_rel {rel:.4e} mean_rel {mean:.4e} ({:.1}x the unmutated {base:.4e}) [{}]",
            m.case,
            m.what,
            rel / base,
            if m.discriminated {
                "must be caught here"
            } else {
                "below the output floor by design"
            }
        );
        if m.discriminated {
            assert!(
                rel > F32_MAX_REL,
                "the gate does not discriminate {:?} on the {} pack: max_rel {rel} vs gate \
                 {F32_MAX_REL}. {}",
                m.what,
                m.case,
                m.note
            );
            assert!(
                rel > base * DISCRIMINATION_FACTOR,
                "{:?} on the {} pack moves the output by max_rel {rel}, only {:.1}x the \
                 unmutated run's own {base} — too close to the noise to call it caught. {}",
                m.what,
                m.case,
                rel / base,
                m.note
            );
        } else {
            assert!(
                rel < F32_MAX_REL,
                "{:?} now moves the {} output by max_rel {rel}, past the {F32_MAX_REL} gate — \
                 this row claimed it was below the numerical floor. Re-measure and promote it. {}",
                m.what,
                m.case,
                m.note
            );
        }
    }
}
