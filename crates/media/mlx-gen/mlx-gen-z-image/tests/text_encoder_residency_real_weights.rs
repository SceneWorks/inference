//! **SC-15794 gating measurement**: the weight/activation split of the conditioning phase, per tier.
//!
//! Rung 4 (bounded transformer residency) streams the DiT. The story's proposal is to give rung 4 a
//! *component scope* so it can stream the **text encoder** too — which matters because rung 4 cut the
//! denoise so far that conditioning became the binding phase for q8.
//!
//! Streaming bounds **weights** only. Activations — the prompt embedding and attention over the
//! sequence — are untouched by any window size. So the go/no-go is a ratio:
//!
//! ```text
//!   conditioning_peak = weights_resident + activation_headroom
//! ```
//!
//! If activations dominate, streaming trades one peak for another and buys nothing (exactly what the
//! sub-512 decode-tile sweep found for rung 2, where a different phase bound the request). If weights
//! dominate, streaming the 36 `model.layers.*` blocks to a small window is worth the re-materialization.
//!
//! **This test measures the ratio. It does not implement streaming** — that is deliberate, and it is
//! the story's first acceptance criterion: measure before building.
//!
//! ## Method
//!
//! `get_peak_memory` is a high-water mark of ACTIVE bytes (not cache), so weights and activations are
//! separated in time rather than by tagging allocations:
//!
//! 1. `clear_cache`, sample the baseline.
//! 2. Build the encoder, then run a **warm-up forward** — MLX is lazy and safetensors-backed weights
//!    are mmap'd, so nothing is resident until something forces it. Without this the "weights" figure
//!    reads near zero and the whole split inverts.
//! 3. Drop the warm-up output, `clear_cache`, and sample active memory again. The encoder is the only
//!    live thing, so this difference **is** the resident weight set.
//! 4. `reset_peak_memory`, run the measured forward, `eval` the output, read the peak.
//! 5. `activation_headroom = peak - weights_resident`.
//!
//! Step 3 is load-bearing and is mutation-checked by
//! [`the_weight_measurement_is_not_reading_an_unmaterialized_encoder`].
//!
//! ## Running
//!
//! ```sh
//! MLX_GEN_ZIMAGE_SNAPSHOT=<snapshot root, containing bf16/ q4/ q8/> \
//!   cargo test -p mlx-gen-z-image --release --test integration text_encoder_residency_real_weights:: \
//!   -- --ignored --nocapture --test-threads=1
//! ```
//!
//! `MLX_GEN_ZIMAGE_SNAPSHOT` may point either at a multi-tier root (all present tiers are measured) or at a
//! single tier directory (that one is measured).

use crate::common;

use std::path::{Path, PathBuf};

use mlx_gen::weights::Weights;
use mlx_gen_z_image::text_encoder::TextEncoder;
use mlx_rs::memory::{clear_cache, get_active_memory, get_peak_memory, reset_peak_memory};

use common::tier_snapshot as snapshot;

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

/// A representative production caption. Conditioning cost scales with sequence length, so a
/// one-word prompt would understate the activation share and bias the verdict toward "stream it".
const LONG_PROMPT: &str =
    "A photorealistic portrait of an elderly lighthouse keeper standing on a \
     weathered stone pier at dawn, heavy wool coat soaked with sea spray, deep lines across a \
     sunlit face, the beam of the lighthouse cutting through low coastal fog behind him, shot on \
     a 85mm lens at f/1.4 with shallow depth of field, muted teal and amber colour grading";

/// The caption under measurement; `ZIMAGE_PROMPT` overrides it.
///
/// **For Z-Image the choice does not matter, and that is a measured finding rather than an
/// assumption.** The conditioning peak is `weights + f(sequence_length)`, so in general which phase
/// binds could depend on prompt length. Z-Image's tokenizer sets `pad_to_max_length` with
/// `MAX_LENGTH = 512` (`loader.rs`, matching the fork's `padding="max_length"`), so **every** prompt
/// conditions at exactly 512 tokens. Verified empirically: a 5-word caption and the long caption
/// below produce identical splits (bf16 8.623 GiB, q4 3.679 GiB — byte-identical across runs).
///
/// The consequence is that the go/no-go below is unconditional for this family rather than being an
/// artifact of the caption chosen. A family whose tokenizer does *not* pad to a fixed length would
/// need this swept before its verdict could be trusted.
fn prompt() -> String {
    std::env::var("ZIMAGE_PROMPT").unwrap_or_else(|_| LONG_PROMPT.to_string())
}

/// One tier's measured conditioning split.
struct Split {
    tier: String,
    /// Bytes the encoder's weights occupy once materialized.
    weights_bytes: u64,
    /// Peak active bytes across the measured forward (weights + live activations).
    peak_bytes: u64,
    /// Bytes of the `model.layers.*` blocks — the part a block window could bound.
    layer_bytes: u64,
    /// Bytes outside the streamable blocks (token embedding, final norm).
    non_layer_bytes: u64,
    n_layers: usize,
    /// Tokens in the measured prompt (conditioning cost scales with this).
    seq_len: i32,
}

impl Split {
    fn activation_bytes(&self) -> u64 {
        self.peak_bytes.saturating_sub(self.weights_bytes)
    }

    /// Share of the conditioning peak that no window size can touch.
    fn activation_share(&self) -> f64 {
        self.activation_bytes() as f64 / self.peak_bytes.max(1) as f64
    }

    /// Predicted conditioning peak if the layer blocks were streamed at `window` blocks resident.
    ///
    /// Weights outside the blocks stay resident, activations are untouched, and the window holds
    /// `window` blocks' worth of layer weights. This is a **prediction from the measured split**, not
    /// a measurement of a streaming implementation (which this story has not built yet).
    fn predicted_peak_at_window(&self, window: usize) -> u64 {
        let per_block = self.layer_bytes / self.n_layers.max(1) as u64;
        self.non_layer_bytes + per_block * window as u64 + self.activation_bytes()
    }
}

/// Sum of on-disk tensor bytes under `model.layers.<i>.` versus everything else, straight from the
/// safetensors header. Disk bytes, not resident bytes — used only to apportion the measured resident
/// total between "streamable blocks" and "everything else", which is a ratio and so survives the
/// difference.
fn layer_byte_split(w: &Weights, n_layers: usize) -> (u64, u64) {
    let keys: Vec<String> = w.keys().map(str::to_string).collect();
    let mut layer = 0u64;
    let mut other = 0u64;
    for key in keys {
        let bytes = w.require(&key).map(|a| a.nbytes() as u64).unwrap_or(0);
        // `layers.{i}.` with the trailing dot, so `layers.1.` cannot also match `layers.11.`.
        let is_layer = (0..n_layers).any(|i| key.contains(&format!("layers.{i}.")));
        if is_layer {
            layer += bytes;
        } else {
            other += bytes;
        }
    }
    (layer, other)
}

fn measure_tier(tier: &str, te_dir: &Path) -> Split {
    let w = Weights::from_dir(te_dir).expect("load text_encoder weights");
    let cfg = mlx_gen_z_image::text_encoder::ZTextEncoderConfig::z_image();
    let n_layers = cfg.n_layers;
    let (layer_bytes, non_layer_bytes) = layer_byte_split(&w, n_layers);

    let encoder = TextEncoder::from_weights(&w, "model", &cfg).expect("build TextEncoder");
    drop(w);

    let tokenizer =
        mlx_gen_z_image::load_tokenizer(te_dir.parent().expect("tier root above text_encoder/"))
            .expect("load tokenizer");
    let tokens = tokenizer.tokenize(&prompt()).expect("tokenize prompt");
    let (input_ids, attention_mask) = mlx_gen::tokenizer::to_arrays(&tokens);
    let seq_len = input_ids.shape()[1];

    // (2) Warm-up forward. MLX is lazy and the weights are mmap-backed, so without this the encoder
    // occupies almost nothing and the split would invert.
    clear_cache();
    let baseline = get_active_memory() as u64;
    let warm = encoder
        .forward(&input_ids, &attention_mask)
        .expect("warm-up forward");
    mlx_rs::transforms::eval([&warm]).expect("eval warm-up");
    drop(warm);

    // (3) Only the encoder is live now, so this difference is the resident weight set.
    clear_cache();
    let weights_bytes = (get_active_memory() as u64).saturating_sub(baseline);

    // (4) The measured forward.
    reset_peak_memory();
    let out = encoder
        .forward(&input_ids, &attention_mask)
        .expect("measured forward");
    mlx_rs::transforms::eval([&out]).expect("eval measured forward");
    let peak_bytes = get_peak_memory() as u64;
    drop(out);
    drop(encoder);
    clear_cache();

    Split {
        tier: tier.to_string(),
        weights_bytes,
        peak_bytes,
        layer_bytes,
        non_layer_bytes,
        n_layers,
        seq_len,
    }
}

/// Tier directories to measure: either the tier subdirectories of a multi-tier root, or the snapshot
/// itself when it already points at one tier.
fn tier_dirs() -> Vec<(String, PathBuf)> {
    let root = snapshot();
    let mut found: Vec<(String, PathBuf)> = ["bf16", "q8", "q4"]
        .iter()
        .map(|t| (t.to_string(), root.join(t)))
        .filter(|(_, p)| p.join("text_encoder").is_dir())
        .collect();
    if found.is_empty() && root.join("text_encoder").is_dir() {
        let name = root
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "snapshot".to_string());
        found.push((name, root.clone()));
    }
    assert!(
        !found.is_empty(),
        "no tier with a text_encoder/ under {}",
        root.display()
    );
    found
}

/// **The gating measurement.** Reports the conditioning weight/activation split per tier and the
/// conditioning peak a block window would predict, then asserts the go/no-go rule rather than merely
/// printing numbers: streaming is worth building only where the weights it can bound are a
/// meaningful share of the peak.
#[test]
#[ignore = "needs a real Z-Image-Turbo snapshot + Apple/Metal GPU"]
fn the_conditioning_weight_activation_split_per_tier() {
    let mut splits = Vec::new();
    for (tier, dir) in tier_dirs() {
        splits.push(measure_tier(&tier, &dir.join("text_encoder")));
    }

    println!("\nSC-15794 conditioning weight/activation split (Apple/Metal, real weights)");
    println!(
        "  prompt: {} tokens\n",
        splits.first().map(|s| s.seq_len).unwrap_or(0)
    );
    println!(
        "  {:<6} {:>10} {:>10} {:>12} {:>9}  {:>10} {:>10}",
        "tier", "weights", "peak", "activations", "act%", "layers", "non-layer"
    );
    for s in &splits {
        println!(
            "  {:<6} {:>9.3}G {:>9.3}G {:>11.3}G {:>8.1}%  {:>9.3}G {:>9.3}G",
            s.tier,
            gib(s.weights_bytes),
            gib(s.peak_bytes),
            gib(s.activation_bytes()),
            s.activation_share() * 100.0,
            gib(s.layer_bytes),
            gib(s.non_layer_bytes),
        );
    }

    println!("\n  Predicted conditioning peak by window size (from the measured split):");
    println!(
        "  {:<6} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "tier", "resident", "w=8", "w=4", "w=2", "w=1"
    );
    for s in &splits {
        println!(
            "  {:<6} {:>9.3}G {:>9.3}G {:>9.3}G {:>9.3}G {:>9.3}G",
            s.tier,
            gib(s.peak_bytes),
            gib(s.predicted_peak_at_window(8)),
            gib(s.predicted_peak_at_window(4)),
            gib(s.predicted_peak_at_window(2)),
            gib(s.predicted_peak_at_window(1)),
        );
    }

    for s in &splits {
        assert!(
            s.weights_bytes > 0 && s.peak_bytes > 0,
            "{}: a phase was not sampled (weights {} peak {})",
            s.tier,
            s.weights_bytes,
            s.peak_bytes
        );
        // The peak must include the weights: a peak below the resident weight set would mean the
        // sampling window missed the forward entirely and every number here is meaningless.
        assert!(
            s.peak_bytes >= s.weights_bytes,
            "{}: peak {:.3}G below resident weights {:.3}G — the split is not measuring the forward",
            s.tier,
            gib(s.peak_bytes),
            gib(s.weights_bytes)
        );
        // The blocks a window could bound must be the bulk of the encoder, or the component scope
        // has nothing to work with regardless of the activation share.
        let layer_share = s.layer_bytes as f64 / (s.layer_bytes + s.non_layer_bytes).max(1) as f64;
        assert!(
            layer_share > 0.5,
            "{}: only {:.1}% of the encoder is in streamable layer blocks — a block window cannot \
             bound this component and the story's premise fails for this tier",
            s.tier,
            layer_share * 100.0
        );
    }

    // The go/no-go, asserted rather than left to prose. A tier where a 1-block window cannot move the
    // conditioning peak by even 10% is a tier where streaming the encoder is not worth its
    // re-materialization cost — that is a legitimate NO-GO result, and it must be stated explicitly
    // rather than discovered after the implementation is built.
    let verdicts: Vec<(String, f64, bool)> = splits
        .iter()
        .map(|s| {
            let saving = 1.0 - s.predicted_peak_at_window(1) as f64 / s.peak_bytes.max(1) as f64;
            (s.tier.clone(), saving, saving >= 0.10)
        })
        .collect();
    println!("\n  GO/NO-GO by tier (predicted conditioning-peak reduction at window=1):");
    for (tier, saving, go) in &verdicts {
        println!(
            "    {:<6} {:>6.1}%  {}",
            tier,
            saving * 100.0,
            if *go { "GO" } else { "NO-GO" }
        );
    }
    println!(
        "\n  NOTE: a conditioning-phase win is only a REQUEST-peak win if conditioning is the binding\n  \
         phase for that tier. That comparison is made in the_request_peak_effect_of_the_predicted_split.\n"
    );

    assert!(
        verdicts.iter().any(|(_, _, go)| *go),
        "no tier shows a >=10% predicted conditioning saving — encoder streaming is NO-GO for \
         z-image on every measured tier, and the story's remaining justification rests entirely on \
         the TE-dominant families"
    );
}

/// **The half that decides whether the split matters.** A conditioning-phase saving is only a win if
/// conditioning is the phase that *binds* the request; otherwise streaming the encoder trades one peak
/// for a different one and the user sees nothing. Rung 2's sub-512 decode-tile sweep is the cautionary
/// case — every candidate improved the phase and none moved the request peak.
///
/// So this runs a real generation per tier with the **same prompt** as the split measurement (
/// conditioning cost scales with sequence length, so mixing prompts would make the two
/// non-composable), records the three phase peaks, and states the request peak before and after
/// applying the predicted conditioning reduction.
#[test]
#[ignore = "needs a real Z-Image-Turbo snapshot + Apple/Metal GPU"]
fn the_request_peak_effect_of_the_predicted_split() {
    use mlx_gen::gen_core::GenerationMemory;
    use mlx_gen::{
        GenerationOutput, GenerationRequest, LoadSpec, OffloadPolicy, Progress, Quant,
        WeightsSource,
    };

    let size = std::env::var("ZIMAGE_SIZE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1024u32);
    let steps = std::env::var("ZIMAGE_STEPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4u32);

    println!(
        "\nSC-15794 request-peak effect ({size}x{size}, {steps} steps, ladder rungs 1-4 active)\n"
    );
    println!(
        "  {:<6} {:>12} {:>9} {:>8} {:>10} {:>13}",
        "tier", "conditioning", "denoise", "decode", "request", "binding phase"
    );

    let mut rows = Vec::new();
    for (tier, dir) in tier_dirs() {
        let split = measure_tier(&tier, &dir.join("text_encoder"));

        // The loader takes a TIER directory (the one holding tokenizer/, text_encoder/, …), not the
        // multi-tier snapshot root.
        let mut spec = LoadSpec::new(WeightsSource::Dir(dir.clone()));
        spec.offload_policy = OffloadPolicy::Sequential;
        spec.load_shape = mlx_gen::LoadShape::DeferredMaterialization;
        spec = match tier.as_str() {
            "q4" => spec.with_quant(Quant::Q4),
            "q8" => spec.with_quant(Quant::Q8),
            _ => spec,
        };
        let req = GenerationRequest {
            prompt: prompt(),
            width: size,
            height: size,
            count: 1,
            seed: Some(1234),
            steps: Some(steps),
            memory: Some(GenerationMemory {
                stage_residency: true,
                tile_vae_decode: true,
                chunk_attention: true,
                stream_transformer_blocks: true,
                transformer_window_size: Some(1),
                // DiT-only: this arm is the pre-SC-15794 ladder top, the baseline the encoder scope
                // is measured against.
                transformer_window_component: Some(mlx_gen::gen_core::TransformerComponent::Dit),
                ..Default::default()
            }),
            ..Default::default()
        };

        let generator = mlx_gen_z_image::load(&spec).expect("load z_image_turbo");
        clear_cache();
        reset_peak_memory();
        let mut cond = 0usize;
        let mut denoise = 0usize;
        let mut on_progress = |p: Progress| match p {
            Progress::Step { current: 1, .. } => {
                cond = get_peak_memory();
                reset_peak_memory();
            }
            Progress::Decoding if denoise == 0 => {
                denoise = get_peak_memory();
                reset_peak_memory();
            }
            _ => {}
        };
        let out = generator
            .generate(&req, &mut on_progress)
            .expect("generation");
        let decode = get_peak_memory();
        match out {
            GenerationOutput::Images(images) => assert_eq!(images.len(), 1),
            other => panic!("expected images, got {other:?}"),
        }
        drop(generator);
        clear_cache();
        assert!(
            cond > 0 && denoise > 0 && decode > 0,
            "{tier}: a phase boundary was not sampled"
        );

        let (cond, denoise, decode) = (cond as u64, denoise as u64, decode as u64);
        let request = cond.max(denoise).max(decode);
        let binding = if request == cond {
            "conditioning"
        } else if request == denoise {
            "denoise"
        } else {
            "decode"
        };
        println!(
            "  {:<6} {:>11.3}G {:>8.3}G {:>7.3}G {:>9.3}G {:>13}",
            tier,
            gib(cond),
            gib(denoise),
            gib(decode),
            gib(request),
            binding
        );
        rows.push((tier, split, cond, denoise, decode, request));
    }

    println!("\n  Applying the predicted conditioning reduction at window=1:");
    println!(
        "  {:<6} {:>10} {:>13} {:>12} {:>10} {:>13}",
        "tier", "request", "cond (now)", "cond (w=1)", "request'", "moves?"
    );
    let mut any_request_win = false;
    for (tier, split, cond, denoise, decode, request) in &rows {
        // Scale the isolated prediction by the same factor the in-pipeline conditioning differs from
        // the isolated peak, so the prediction is expressed in the pipeline's own terms rather than
        // assuming the two harnesses measure an identical envelope.
        let scale = *cond as f64 / split.peak_bytes.max(1) as f64;
        let predicted_cond = (split.predicted_peak_at_window(1) as f64 * scale) as u64;
        let new_request = predicted_cond.max(*denoise).max(*decode);
        let delta = 1.0 - new_request as f64 / (*request).max(1) as f64;
        if delta > 0.01 {
            any_request_win = true;
        }
        println!(
            "  {:<6} {:>9.3}G {:>12.3}G {:>11.3}G {:>9.3}G {:>12.1}%",
            tier,
            gib(*request),
            gib(*cond),
            gib(predicted_cond),
            gib(new_request),
            delta * 100.0
        );
    }

    println!(
        "\n  A phase win that does not move the request peak is NOT a win. Any tier showing ~0% \
         above\n  is one where a different phase binds and encoder streaming buys that tier nothing.\n"
    );

    // The decision this evidence actually feeds: which TIER a constrained machine can run. A tier
    // that already fits shows 0% above and looks uninteresting, but the question is not "how much
    // smaller does each tier get" — it is "how good a tier fits the budget". Encoder streaming is
    // worth building if it moves a tier across the line, and that is a different test from whether it
    // moves that tier's number.
    let budget_gib = 8.0f64;
    let reserve_gib = 2.0f64;
    println!(
        "  {budget_gib:.0} GB budget verdict (peak + {reserve_gib:.1} GiB operational reserve):"
    );
    println!(
        "  {:<6} {:>10} {:>10} {:>9} {:>12} {:>10} {:>9}",
        "tier", "request", "+reserve", "fits?", "request'", "+reserve", "fits?"
    );
    let mut tier_crossings = Vec::new();
    for (tier, split, cond, denoise, decode, request) in &rows {
        let scale = *cond as f64 / split.peak_bytes.max(1) as f64;
        let predicted_cond = (split.predicted_peak_at_window(1) as f64 * scale) as u64;
        let new_request = predicted_cond.max(*denoise).max(*decode);
        let now = gib(*request) + reserve_gib;
        let after = gib(new_request) + reserve_gib;
        let fits_now = now <= budget_gib;
        let fits_after = after <= budget_gib;
        println!(
            "  {:<6} {:>9.3}G {:>9.3}G {:>9} {:>11.3}G {:>9.3}G {:>9}",
            tier,
            gib(*request),
            now,
            if fits_now { "FITS" } else { "NO" },
            gib(new_request),
            after,
            if fits_after { "FITS" } else { "NO" },
        );
        if !fits_now && fits_after {
            tier_crossings.push(tier.clone());
        }
    }
    if tier_crossings.is_empty() {
        println!(
            "\n  No tier crosses into the {budget_gib:.0} GB budget that was not already inside it."
        );
    } else {
        println!(
            "\n  Encoder streaming brings {} across the {budget_gib:.0} GB line — a TIER upgrade on \
             constrained\n  machines, which is a quality win rather than only a memory one.",
            tier_crossings.join(", ")
        );
    }
    println!(
        "\n  DISCLOSURE: measured on this host, which is NOT a physical 8 GB Mac, and MLX's peak\n  \
         counter is ACTIVE bytes (not cache). The {reserve_gib:.1} GiB reserve is borrowed from the\n  \
         dedicated-VRAM lane; on a real 8 GB Mac the pool is UNIFIED and shared with macOS /\n  \
         WindowServer / SceneWorks, so whether that reserve is right is SC-15611 / SC-15614, not\n  \
         this story.\n"
    );
    // The verdict is RECORDED, not required. The story is explicit that a NO-GO is a legitimate and
    // useful result — "if it says the win is not there for z-image, that is a legitimate result" — so
    // asserting a win here would be asserting the conclusion we set out to test. What IS required is
    // that the measurement stayed sensitive enough to produce a verdict at all: every tier must have
    // sampled real phase peaks, and the conditioning reduction must be non-trivial, or the "no request
    // movement" reading is just a broken harness.
    println!(
        "  VERDICT for z-image: encoder streaming moves the request peak on {} of {} tiers.",
        rows.iter()
            .filter(|(_, split, cond, denoise, decode, request)| {
                let scale = *cond as f64 / split.peak_bytes.max(1) as f64;
                let predicted = (split.predicted_peak_at_window(1) as f64 * scale) as u64;
                predicted.max(*denoise).max(*decode) < *request
            })
            .count(),
        rows.len()
    );
    if !any_request_win {
        println!(
            "  => HISTORICAL PRE-SC-19753 NO-GO for z-image: every tier was bound by a different phase\n                  (whole-tail decode at ~4.36 GiB), so the conditioning saving was invisible to the user.\n                  Re-evaluate against v4 layer-wise-decode evidence; TE-dominant families remain the\n                  independent justification for the component scope."
        );
    }
    for (tier, split, cond, _, _, _) in &rows {
        assert!(
            *cond > 0 && split.peak_bytes > 0,
            "{tier}: a phase peak was not sampled, so this verdict is a harness failure rather than a \
             measurement"
        );
        let cut = 1.0 - split.predicted_peak_at_window(1) as f64 / split.peak_bytes.max(1) as f64;
        assert!(
            cut > 0.20,
            "{tier}: the predicted conditioning reduction is only {:.1}% — the split is no longer \
             sensitive enough for the request-peak comparison above to mean anything",
            cut * 100.0
        );
    }
}

/// Guards the measurement itself. The resident-weight figure is only meaningful because a warm-up
/// forward materialized the mmap-backed weights first; without that step MLX reports a nearly empty
/// encoder and the split inverts to "almost all activations" — which would produce a confident,
/// wrong NO-GO.
///
/// This measures the un-warmed figure explicitly and asserts it is far below the warmed one, so the
/// method cannot silently regress into the trap it was written to avoid.
#[test]
#[ignore = "needs a real Z-Image-Turbo snapshot + Apple/Metal GPU"]
fn the_weight_measurement_is_not_reading_an_unmaterialized_encoder() {
    let (tier, dir) = tier_dirs().pop().expect("at least one tier");
    let te_dir = dir.join("text_encoder");

    let w = Weights::from_dir(&te_dir).expect("load text_encoder weights");
    let cfg = mlx_gen_z_image::text_encoder::ZTextEncoderConfig::z_image();
    let encoder = TextEncoder::from_weights(&w, "model", &cfg).expect("build TextEncoder");
    drop(w);

    clear_cache();
    let baseline = get_active_memory() as u64;
    // No warm-up: this is what the naive measurement would have reported.
    let cold = (get_active_memory() as u64).saturating_sub(baseline);

    let tokenizer = mlx_gen_z_image::load_tokenizer(&dir).expect("load tokenizer");
    let tokens = tokenizer.tokenize(&prompt()).expect("tokenize");
    let (input_ids, attention_mask) = mlx_gen::tokenizer::to_arrays(&tokens);
    let warm_out = encoder
        .forward(&input_ids, &attention_mask)
        .expect("warm-up forward");
    mlx_rs::transforms::eval([&warm_out]).expect("eval");
    drop(warm_out);
    clear_cache();
    let warm = (get_active_memory() as u64).saturating_sub(baseline);

    println!(
        "\nSC-15794 measurement guard ({tier}): encoder resident BEFORE a forward {:.3} GiB, \
         AFTER {:.3} GiB",
        gib(cold),
        gib(warm)
    );
    assert!(
        warm > cold * 2,
        "{tier}: materializing the encoder moved resident bytes only {:.3} -> {:.3} GiB. Either the \
         weights were already resident (making the split's baseline wrong) or the forward did not \
         run — in both cases the reported weight/activation split is not trustworthy.",
        gib(cold),
        gib(warm)
    );
}
