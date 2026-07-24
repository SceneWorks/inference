//! sc-14038 — **real-weight parity** for the Qwen3-VL-4B text encoder against the vendored torch
//! reference's CPU boundary golden (`tools/dump_mage_flow_golden.py --stage te`, sc-14036).
//!
//! `#[ignore]` — needs the 8.875 GB `text_encoder/` snapshot **and** the golden, which is
//! gitignored (`crates/media/mlx-gen/tools/golden/README.md`):
//!
//! ```sh
//! MAGE_SNAPSHOT=/path/to/models--microsoft--Mage-Flow/snapshots/<rev> \
//! MAGE_FLOW_TE_GOLDEN=/path/to/tools/golden/mage_flow_te_golden.safetensors \
//!   cargo test -p mlx-gen-mage --release --test te_parity_real_weights -- --ignored --nocapture
//! ```
//!
//! Both are **caller-provisioned local paths** — this crate never derives a cache location
//! (`scripts/check-workspace.py`, the epic-13657 self-fetch boundary).
//!
//! ## What the golden gives us, and why that matters
//!
//! The dump captures the LM's `last_hidden_state` through a forward hook *before* the reference
//! drops the system prompt, so `gen_hidden_full` (94 × 2560, the packed positive **and** negative
//! sequences) and `gen_txt` / `neg_txt` (the post-drop slices) are independent oracles. A port can
//! therefore tell "wrong hidden layer / missing final RMSNorm" apart from "wrong `drop_idx`" —
//! [`discrimination_wrong_layer_and_wrong_drop_idx_both_fail`] makes that separation executable
//! rather than asserted.
//!
//! ## Tolerance, not equality
//!
//! The reference runs the encoder in **bf16** (`pipeline.py:754` `model.txt_enc.to(torch.bfloat16)`)
//! while this port runs f32 activations over the bf16 weight store, the convention every Qwen3
//! text encoder in this workspace uses. Thirty-six decoder layers accumulate that difference. The
//! reference's own device spread is the calibration: the *same* code on MPS instead of CPU moves
//! this tensor by `mean_rel ≈ 2.7e-2` (`_vendor/MAGE_FLOW_GAPS.md` GAP 1). This port measures
//! **2.18e-2** against the CPU golden — i.e. it already sits at the oracle's own noise floor — and
//! the gates below are still 2–3 orders of magnitude away from every structural mistake, which is
//! what makes them a gate rather than a rubber stamp. See the tolerance block for the full
//! sensitivity table and for the one mutation the golden provably cannot separate.

use std::path::PathBuf;

use mlx_rs::{Array, Dtype};

use mlx_gen::weights::Weights;
use mlx_gen_mage::config::QwenVlTextConfig;
use mlx_gen_mage::text_encoder::{self, MageTextEncoder, PromptKind, Qwen3VlTextEncoder};

/// The fixed prompt the golden was dumped with (`dump_mage_flow_golden.py:103-106`).
const PROMPT: &str = "a calico kitten sitting on a wooden windowsill beside a blue ceramic mug";
/// The default negative prompt (`pipeline.py` `_as_list(neg_prompts, " ", n)`).
const NEGATIVE: &str = " ";
/// The edit instruction the golden was dumped with.
const EDIT_INSTRUCTION: &str = "Replace the background with a field of sunflowers";

// ── Tolerance ────────────────────────────────────────────────────────────────────────────────
//
// **Measured**, on the real 4.1B checkpoint (Apple M-series, MLX Metal) against the committed CPU
// golden: `max_abs 1.523`, `mean_rel 2.18e-2` on a tensor whose own peak magnitude is 113.
//
// Why that is the *expected* distance rather than a defect. The reference runs the encoder in
// bf16 (`pipeline.py:754`); this port runs f32 activations over the bf16 weight store. bf16 keeps
// 8 mantissa bits, so each layer contributes ~2⁻⁹ ≈ 2e-3 of relative rounding, and 36 residual
// layers accumulate it. The reference's own CPU↔MPS spread — two *independent* bf16 runs of the
// same code — is `mean_rel ≈ 2.7e-2` (`_vendor/MAGE_FLOW_GAPS.md` GAP 1); two independent errors
// of size ε differ by about ε·√2, which puts a single bf16 run at ≈1.9e-2 from the exact value.
// This port measures 2.18e-2 from the CPU golden, i.e. almost exactly the golden's own noise
// floor. An f32 port therefore *cannot* do better against a bf16 oracle, and running the port in
// bf16 instead would make it worse (two independent errors rather than one).
//
// Measured sensitivity of this exact comparison, all against `gen_hidden_full` (94 × 2560):
//
// | candidate                                   | max_abs   | mean_rel |
// | ------------------------------------------- | --------- | -------- |
// | **this port** (final POST-norm, drop 34)     | **1.523** | 2.18e-2  |
// | penultimate layer (the z-image convention)   | 10503.13  | 1.84e0   |
// | final layer, final RMSNorm omitted           | 4309.12   | 1.92e0   |
// | `drop_idx = 0` (vs `gen_txt`)                | 140.23    | 1.08e0   |
// | `rope_theta = 1e6` (the Z-Image base)        | 7.080     | 6.80e-2  |
// | `q_norm`/`k_norm` eps `1e-5` (mlx default)   | 2.300     | 2.50e-2  |
//
// The gates below sit ~2× above the measurement (so Metal kernel selection on another machine
// cannot red the lane) and above the reference's own 2.7e-2 device spread (a gate tighter than the
// oracle's own reproducibility would be measuring the machine, not the port) — while still
// rejecting every structural mistake by 2–3 orders of magnitude and the Z-Image rotary base by
// ~2×. The one mutation they do NOT separate is the QK-norm epsilon, which is guarded instead by
// [`text_encoder::verify_text_config`]; see
// `subtly_wrong_hyperparameters_are_caught_by_the_gate_or_by_the_config_guard`, which asserts both
// halves of that split so it cannot rot into an unnoticed gap.

/// Largest permitted absolute difference on any element of the 2560-wide hidden state (measured
/// 1.523; the tensor itself reaches |113|, so this is ~2.7% of peak magnitude).
const MAX_ABS_TOL: f32 = 3.0;
/// Largest permitted `mean|Δ| / mean|golden|` (measured 2.18e-2).
const MEAN_REL_TOL: f32 = 3.5e-2;

fn snapshot() -> PathBuf {
    PathBuf::from(
        std::env::var("MAGE_SNAPSHOT")
            .expect("set MAGE_SNAPSHOT to a microsoft/Mage-Flow* snapshot root"),
    )
}

fn golden() -> Weights {
    let path = std::env::var("MAGE_FLOW_TE_GOLDEN").unwrap_or_else(|_| {
        format!(
            "{}/../tools/golden/mage_flow_te_golden.safetensors",
            env!("CARGO_MANIFEST_DIR")
        )
    });
    Weights::from_file(&path).unwrap_or_else(|e| {
        panic!(
            "golden {path}: {e} — the goldens are gitignored; point MAGE_FLOW_TE_GOLDEN at a \
             `dump_mage_flow_golden.py --stage te` bundle (MAGE_DEVICE=cpu is mandatory, sc-14250)"
        )
    })
}

fn f32_of(w: &Weights, key: &str) -> Array {
    w.require(key)
        .unwrap_or_else(|e| panic!("golden key {key}: {e}"))
        .as_dtype(Dtype::Float32)
        .unwrap()
}

fn i32_vec(w: &Weights, key: &str) -> Vec<i32> {
    let a = w
        .require(key)
        .unwrap_or_else(|e| panic!("golden key {key}: {e}"))
        .as_dtype(Dtype::Int32)
        .unwrap();
    a.as_slice::<i32>().to_vec()
}

/// `(max|Δ|, mean|Δ| / mean|want|)`. The second is a scale-normalised mean absolute error, which
/// is the shape of the `mean_rel` number the P0 story reports for the reference's own CPU↔MPS
/// spread — so the two are directly comparable.
fn stats(got: &Array, want: &Array) -> (f32, f32) {
    assert_eq!(got.shape(), want.shape(), "shape mismatch");
    let (g, w) = (got.as_slice::<f32>(), want.as_slice::<f32>());
    let mut max_abs = 0f32;
    let mut sum_abs = 0f64;
    let mut sum_ref = 0f64;
    for (x, y) in g.iter().zip(w) {
        let d = (x - y).abs();
        max_abs = max_abs.max(d);
        sum_abs += d as f64;
        sum_ref += y.abs() as f64;
    }
    (max_abs, (sum_abs / sum_ref.max(f64::MIN_POSITIVE)) as f32)
}

fn report(label: &str, got: &Array, want: &Array) -> (f32, f32) {
    let (max_abs, mean_rel) = stats(got, want);
    println!(
        "  {label:<24} shape {:?}  max_abs {max_abs:.5}  mean_rel {mean_rel:.3e}",
        got.shape()
    );
    (max_abs, mean_rel)
}

/// `true` when a candidate would pass the parity gate.
fn passes(max_abs: f32, mean_rel: f32) -> bool {
    max_abs <= MAX_ABS_TOL && mean_rel <= MEAN_REL_TOL
}

fn assert_within(label: &str, got: &Array, want: &Array) {
    let (max_abs, mean_rel) = report(label, got, want);
    assert!(
        max_abs <= MAX_ABS_TOL,
        "{label}: max_abs {max_abs} exceeds {MAX_ABS_TOL}"
    );
    assert!(
        mean_rel <= MEAN_REL_TOL,
        "{label}: mean_rel {mean_rel:e} exceeds {MEAN_REL_TOL:e}"
    );
}

/// Load the encoder once per test binary invocation. Each test that needs it pays the load; the
/// suite is `#[ignore]`d and run deliberately, and `RUST_TEST_THREADS=1` (forced repo-wide) means
/// a shared static would not buy concurrency anyway.
fn load() -> MageTextEncoder {
    text_encoder::load(snapshot()).expect("load text_encoder/")
}

/// The raw `text_encoder/` weight map, so a test can build several encoders that differ ONLY in a
/// hyperparameter. mlx arrays are reference-counted, so the variants share one 8.875 GB store.
fn lm_weights() -> Weights {
    Weights::from_dir(snapshot().join(text_encoder::COMPONENT_DIR)).expect("text_encoder weights")
}

fn lm_with(w: &Weights, eps: f32, rope_theta: f64) -> Qwen3VlTextEncoder {
    Qwen3VlTextEncoder::from_weights(
        w,
        text_encoder::LM_PREFIX,
        &QwenVlTextConfig::mage_flow(),
        eps,
        rope_theta,
    )
    .expect("build encoder")
}

/// The packed positive+negative pre-drop hidden state, as f32.
fn packed_hidden(lm: &Qwen3VlTextEncoder, pos: &[i32], neg: &[i32]) -> Array {
    let mut ids = pos.to_vec();
    ids.extend_from_slice(neg);
    let cu = text_encoder::cu_seqlens_from_lens(&[pos.len(), neg.len()]);
    lm.forward_packed(&ids, &cu)
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap()
}

/// The templated prompt must tokenize to the reference's `input_ids` **exactly**. A one-token
/// transcription slip in the system prefix would shift `drop_idx` and corrupt every conditioning
/// tensor while still producing a plausible-looking result.
#[test]
#[ignore = "needs MAGE_SNAPSHOT + the gitignored TE golden"]
fn tokenizer_reproduces_the_reference_input_ids() {
    let g = golden();
    let te = MageTextEncoder::new(
        text_encoder::load_tokenizer(snapshot()).expect("tokenizer.json"),
        // The tokenizer half needs no weights; build the LM lazily only where it is used.
        text_encoder::load_lm(snapshot()).expect("text_encoder weights"),
    );

    let ids = te.token_ids(PROMPT, PromptKind::Gen).unwrap();
    let want = i32_vec(&g, "gen_input_ids");
    assert_eq!(ids.len(), want.len(), "templated prompt token count");
    assert_eq!(ids, want, "templated prompt ids diverge from the reference");

    // The system prefix must be exactly `drop_idx` tokens — the invariant `drop_idx` encodes.
    let prefix_only = te
        .tokenizer()
        .encode_ids(PromptKind::Gen.template().split("{}").next().unwrap(), true)
        .unwrap();
    assert_eq!(
        prefix_only.len(),
        PromptKind::Gen.drop_idx(),
        "the generation system prefix must tokenize to exactly drop_idx tokens"
    );
    assert_eq!(
        &ids[..PromptKind::Gen.drop_idx()],
        prefix_only.as_slice(),
        "the dropped slice is not the system prefix"
    );
}

/// **The DoD gate.** The packed positive+negative encode must match the torch golden — both the
/// pre-drop `gen_hidden_full` (which pins the layer + final norm) and the post-drop `gen_txt` /
/// `neg_txt` (which pin `drop_idx` and per-segment isolation).
#[test]
#[ignore = "needs MAGE_SNAPSHOT + the gitignored TE golden"]
fn conditioning_matches_the_torch_golden() {
    let g = golden();
    let te = load();

    let pos_ids = te.token_ids(PROMPT, PromptKind::Gen).unwrap();
    let neg_ids = te.token_ids(NEGATIVE, PromptKind::Gen).unwrap();
    let mut packed = pos_ids.clone();
    packed.extend_from_slice(&neg_ids);
    let cu = text_encoder::cu_seqlens_from_lens(&[pos_ids.len(), neg_ids.len()]);

    println!("packed segments: {} + {}", pos_ids.len(), neg_ids.len());

    // (1) The full pre-drop packed hidden state — the layer/final-norm oracle.
    let hidden = te
        .lm()
        .forward_packed(&packed, &cu)
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap();
    assert_within("gen_hidden_full", &hidden, &f32_of(&g, "gen_hidden_full"));

    // (2) The conditioning the DiT actually consumes, through the public encode path.
    let cond = te.encode(&[PROMPT, NEGATIVE], PromptKind::Gen).unwrap();
    assert_eq!(
        cond.seq_lens,
        vec![
            i32_vec(&g, "gen_txt_len")[0] as usize,
            i32_vec(&g, "neg_txt_len")[0] as usize
        ],
        "post-drop segment lengths"
    );
    assert_within(
        "gen_txt",
        &cond.segment(0).unwrap().as_dtype(Dtype::Float32).unwrap(),
        &f32_of(&g, "gen_txt"),
    );
    // `neg_txt` is the SECOND packed segment: a varlen implementation that leaked across segment
    // boundaries would corrupt it while leaving `gen_txt` intact. It is also the exact tensor the
    // MPS golden-dump corruption (sc-14250) showed up in.
    assert_within(
        "neg_txt",
        &cond.segment(1).unwrap().as_dtype(Dtype::Float32).unwrap(),
        &f32_of(&g, "neg_txt"),
    );
}

/// **Discrimination.** The parity gate above must reject the two mistakes this port exists to
/// correct. Both are computed here from the *same* weights and the *same* forward, so a pass is
/// evidence about the gate, not about the fixture.
///
/// * wrong layer — the `mlx-gen-z-image` penultimate convention, and the final layer without the
///   final RMSNorm;
/// * wrong `drop_idx` — keeping the system prompt (`drop_idx = 0`) instead of dropping 34.
#[test]
#[ignore = "needs MAGE_SNAPSHOT + the gitignored TE golden"]
fn discrimination_wrong_layer_and_wrong_drop_idx_both_fail() {
    let g = golden();
    let te = load();
    let want_full = f32_of(&g, "gen_hidden_full");
    let want_txt = f32_of(&g, "gen_txt");

    let pos_ids = te.token_ids(PROMPT, PromptKind::Gen).unwrap();
    let neg_ids = te.token_ids(NEGATIVE, PromptKind::Gen).unwrap();

    // Re-run the stack by hand so the penultimate and pre-norm states are observable. Segments are
    // independent (see `Qwen3VlTextEncoder::forward_packed`), so encoding each and concatenating
    // reproduces the packed tensor.
    let mut penult_parts = Vec::new();
    let mut prenorm_parts = Vec::new();
    let mut post_parts = Vec::new();
    for ids in [&pos_ids, &neg_ids] {
        let n = ids.len() as i32;
        let ids_arr = Array::from_slice(ids, &[1, n]);
        let mut h = te.lm().embed(&ids_arr).unwrap();
        let pos = text_encoder::MRopePositions::text(ids.len());
        let (cos, sin) =
            text_encoder::mrope_cos_sin(&pos, 128, 5_000_000.0, [24, 20, 20], h.dtype()).unwrap();
        let ones = vec![1i32; ids.len()];
        let mask = mlx_gen::nn::build_mask(&Array::from_slice(&ones, &[1, n]), 1, n).unwrap();

        let mut penultimate = h.clone();
        for layer in te.lm().layers() {
            let next = layer.forward(&h, &cos, &sin, &mask).unwrap();
            penultimate = std::mem::replace(&mut h, next);
        }
        let flat = |x: &Array| {
            x.reshape(&[n, -1])
                .unwrap()
                .as_dtype(Dtype::Float32)
                .unwrap()
        };
        penult_parts.push(flat(&penultimate));
        prenorm_parts.push(flat(&h));
        post_parts.push(flat(&te.lm().final_norm(&h).unwrap()));
    }
    let cat = |parts: &[Array]| {
        let refs: Vec<&Array> = parts.iter().collect();
        mlx_rs::ops::concatenate_axis(&refs, 0).unwrap()
    };

    println!("wrong-answer distances vs the golden `gen_hidden_full`:");
    let (penult_abs, penult_rel) = report("penultimate", &cat(&penult_parts), &want_full);
    let (prenorm_abs, prenorm_rel) = report("final pre-norm", &cat(&prenorm_parts), &want_full);
    let (ok_abs, ok_rel) = report("final POST-norm", &cat(&post_parts), &want_full);

    assert!(
        !passes(penult_abs, penult_rel),
        "the penultimate layer passes the parity gate ({penult_abs}/{penult_rel:e}) — the gate \
         does not discriminate the z-image convention"
    );
    assert!(
        !passes(prenorm_abs, prenorm_rel),
        "the final layer WITHOUT the final RMSNorm passes the parity gate \
         ({prenorm_abs}/{prenorm_rel:e})"
    );
    assert!(
        passes(ok_abs, ok_rel),
        "the correct candidate must still pass ({ok_abs}/{ok_rel:e})"
    );
    // The wrong answers are not marginal: they are orders of magnitude out.
    assert!(
        penult_abs > 100.0 * MAX_ABS_TOL && prenorm_abs > 100.0 * MAX_ABS_TOL,
        "wrong-layer separation collapsed: penultimate {penult_abs}, pre-norm {prenorm_abs}"
    );

    // Wrong `drop_idx`: keep the system prompt. The lengths differ, so compare the leading rows
    // the DiT would actually have consumed.
    let mut packed = pos_ids.clone();
    packed.extend_from_slice(&neg_ids);
    let cu = text_encoder::cu_seqlens_from_lens(&[pos_ids.len(), neg_ids.len()]);
    let undropped = te
        .lm()
        .forward_packed(&packed, &cu)
        .unwrap()
        .as_dtype(Dtype::Float32)
        .unwrap();
    let keep = want_txt.shape()[0];
    let no_drop = undropped.split_axis(&[keep], 0).unwrap().remove(0);
    let (drop0_abs, drop0_rel) = report("drop_idx = 0", &no_drop, &want_txt);
    assert!(
        !passes(drop0_abs, drop0_rel),
        "drop_idx = 0 passes the parity gate ({drop0_abs}/{drop0_rel:e}) — the gate does not \
         discriminate the system-prompt drop"
    );

    // …and the correct drop does pass, from the same tensor.
    let cond = te.encode(&[PROMPT, NEGATIVE], PromptKind::Gen).unwrap();
    let (ok_drop_abs, ok_drop_rel) = report(
        "drop_idx = 34",
        &cond.segment(0).unwrap().as_dtype(Dtype::Float32).unwrap(),
        &want_txt,
    );
    assert!(passes(ok_drop_abs, ok_drop_rel));
}

/// **Discrimination, subtle edition — and an honest limit of the golden.**
///
/// The wrong *layers* are off by thousands, so almost any gate catches them. The dangerous
/// mistakes are the single scalars a port could inherit from the `mlx-gen-z-image` sibling, which
/// announce themselves with no shape mismatch and no missing key:
///
/// * `rope_theta = 1e6` (the Z-Image value) instead of Qwen3-VL's 5e6;
/// * `q_norm`/`k_norm` at mlx's default `eps = 1e-5` instead of the config's `rms_norm_eps = 1e-6`
///   — the sibling uses the default *deliberately*, because its fork builds those norms without an
///   explicit eps.
///
/// This test measures both against the same golden and asserts what is actually true of each:
/// `rope_theta` is caught by the parity gate; **the QK-norm eps is not**, because at `head_dim`
/// 128 the mean square of a normalised head is many orders of magnitude above either epsilon, so
/// the two values are nearly the same function and the difference lands inside the golden's own
/// bf16 noise. Rather than pretend otherwise — or tighten the tolerance to within 1.3× of the
/// measurement, which would just make the lane machine-dependent — the eps is pinned by the
/// **config conformance guard**, which this test also exercises so the compensating control cannot
/// silently disappear.
#[test]
#[ignore = "needs MAGE_SNAPSHOT + the gitignored TE golden"]
fn subtly_wrong_hyperparameters_are_caught_by_the_gate_or_by_the_config_guard() {
    let g = golden();
    let want_full = f32_of(&g, "gen_hidden_full");
    let tok = text_encoder::load_tokenizer(snapshot()).expect("tokenizer.json");
    let w = lm_weights();

    let render = |body: &str| {
        let mut ids = tok.encode_ids(&PromptKind::Gen.render(body), true).unwrap();
        ids.truncate(PromptKind::Gen.max_prompt_tokens());
        ids
    };
    let (pos_ids, neg_ids) = (render(PROMPT), render(NEGATIVE));

    let measure = |label: &str, eps: f32, theta: f64| {
        let variant = lm_with(&w, eps, theta);
        report(
            label,
            &packed_hidden(&variant, &pos_ids, &neg_ids),
            &want_full,
        )
    };

    println!("hyperparameter mutations vs the golden `gen_hidden_full`:");
    let (ok_abs, ok_rel) = measure("eps 1e-6 / theta 5e6", 1e-6, 5_000_000.0);
    let (eps_abs, eps_rel) = measure("QK-norm eps 1e-5", 1e-5, 5_000_000.0);
    let (theta_abs, theta_rel) = measure("rope_theta 1e6", 1e-6, 1_000_000.0);

    assert!(
        passes(ok_abs, ok_rel),
        "the production hyperparameters must pass ({ok_abs}/{ok_rel:e})"
    );

    // `rope_theta` changes every rotation angle — the gate catches it outright.
    assert!(
        !passes(theta_abs, theta_rel),
        "rope_theta 1e6 passes the parity gate ({theta_abs}/{theta_rel:e}) — the tolerance is too \
         loose to catch the Z-Image rotary base"
    );

    // The QK-norm eps DOES move the tensor, but not out of the noise. Both halves are asserted:
    // if the mutation ever became a no-op the probe would be worthless, and if it ever became
    // separable this assertion fires and the right response is to tighten `MAX_ABS_TOL` and delete
    // this branch.
    assert!(
        eps_abs > ok_abs,
        "QK-norm eps 1e-5 is a complete no-op ({eps_abs} vs {ok_abs}) — the probe proves nothing"
    );
    assert!(
        passes(eps_abs, eps_rel),
        "QK-norm eps 1e-5 now FAILS the parity gate ({eps_abs}/{eps_rel:e}). That is good news: \
         tighten MAX_ABS_TOL/MEAN_REL_TOL to gate it directly and remove this branch."
    );

    // …so the compensating control is the config guard, which rejects a checkpoint declaring the
    // sibling's epsilon before a single weight is read.
    let published = std::fs::read_to_string(
        snapshot()
            .join(text_encoder::COMPONENT_DIR)
            .join("config.json"),
    )
    .expect("published text_encoder/config.json");
    text_encoder::verify_text_config(&published).expect("the published config must verify");

    let mut v: serde_json::Value = serde_json::from_str(&published).unwrap();
    v["text_config"]["rms_norm_eps"] = serde_json::json!(1e-5);
    assert!(
        text_encoder::verify_text_config(&v.to_string()).is_err(),
        "the config guard is the ONLY thing standing between this port and the sibling's \
         QK-norm epsilon, and it did not fire"
    );
}

/// The **edit** template's token plumbing, up to (not including) the vision tower — the seam
/// sc-14048 builds on. Reconstructing the reference's `edit_input_ids` from the pinned edit
/// template, [`edit_body`](text_encoder::edit_body) and the merged-token count derived from the
/// golden's own `image_grid_thw` proves the text half is right before any vision code exists.
///
/// The reference reaches these ids through `AutoProcessor`, which expands the single
/// `<|image_pad|>` placeholder to one token per merged patch — `prod(grid) / spatial_merge²`.
#[test]
#[ignore = "needs MAGE_SNAPSHOT + the gitignored TE golden"]
fn edit_template_reproduces_the_reference_edit_input_ids() {
    let g = golden();
    let tok = text_encoder::load_tokenizer(snapshot()).expect("tokenizer.json");

    let grid = i32_vec(&g, "edit_image_grid_thw");
    assert_eq!(grid.len(), 3, "one reference image");
    // `spatial_merge_size` is 2 for the whole Qwen3-VL family (published `vision_config`).
    let merged = (grid[0] * grid[1] * grid[2] / 4) as usize;
    println!("edit grid {grid:?} -> {merged} merged vision tokens");

    let body = text_encoder::edit_body(EDIT_INSTRUCTION, 1);
    // Expand the placeholder the way the processor does.
    let expanded = body.replace("<|image_pad|>", &"<|image_pad|>".repeat(merged));
    let text = PromptKind::Edit.render(&expanded);
    let ids = tok.encode_ids(&text, true).unwrap();

    let want = i32_vec(&g, "edit_input_ids");
    assert_eq!(ids.len(), want.len(), "edit sequence length");
    assert_eq!(ids, want, "edit input ids diverge from the reference");

    // …and the edit drop is 64, leaving the length the golden records.
    assert_eq!(PromptKind::Edit.drop_idx(), 64);
    assert_eq!(
        ids.len() - PromptKind::Edit.drop_idx(),
        i32_vec(&g, "edit_txt_len")[0] as usize
    );
}
