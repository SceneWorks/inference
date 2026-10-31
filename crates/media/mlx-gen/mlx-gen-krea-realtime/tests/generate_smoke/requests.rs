//! The weights-free request gates and the request builders / real-weight driver they share.

use super::*;

// ---------------------------------------------------------------------------------------------
// Weights-free gate (runs in CI)
// ---------------------------------------------------------------------------------------------

/// **sc-15325 regression guard — the tiled decode must never starve the temporal receptive field.**
///
/// Weights-free arithmetic over the *product* policy (`decode_tiling`), at every bucket the old
/// policy collapsed at, under a budget too small for a full-frame single pass — i.e. exactly the
/// regime the defect lived in. The old policy emitted an 8-output-frame window = **2 latent frames**
/// (overlap 1, the clamp maximum there) at all of these, so this test is red on it by construction;
/// it is a property gate, not a snapshot of today's numbers.
///
/// The invariant: *if* a temporal tile is emitted, it spans ≥ `MIN_TEMPORAL_TILE_LATENT_FRAMES`
/// latent frames with ≥ `MIN_TEMPORAL_TILE_LATENT_OVERLAP` latent frames of blend. A plan with no
/// temporal tiling is trivially fine — the decoder sees the whole sequence.
#[test]
fn decode_tiling_never_starves_the_temporal_receptive_field() {
    use mlx_gen::tiling::{MIN_TEMPORAL_TILE_LATENT_FRAMES, MIN_TEMPORAL_TILE_LATENT_OVERLAP};
    const TEMPORAL_SCALE: i32 = 4; // VaeTiling::WAN

    // Pin the budget so this gates the policy, not the host's free memory.
    std::env::set_var("WAN_VAE_BUDGET_GIB", "12");
    for (w, h) in [
        (832usize, 480usize),
        (640, 384),
        (512, 512),
        (768, 512),
        (1280, 720),
        (480, 832),
        (512, 384),
    ] {
        let cfg = decode_tiling(h, w, 84)
            .unwrap_or_else(|e| panic!("{w}x{h} must stay decodable within 12 GiB: {e}"))
            .unwrap_or_else(|| panic!("{w}x{h}/84f cannot fit a 12 GiB single pass"));
        let Some(t) = cfg.temporal else { continue };
        let lat_tile = t.tile_frames / TEMPORAL_SCALE;
        let lat_over = (t.overlap_frames / TEMPORAL_SCALE).min(lat_tile - 1);
        assert!(
            lat_tile >= MIN_TEMPORAL_TILE_LATENT_FRAMES,
            "{w}x{h}: temporal tile {} output frames = {lat_tile} LATENT frames, under the \
             {MIN_TEMPORAL_TILE_LATENT_FRAMES}-frame receptive-field floor — this is the sc-15325 \
             defect (2 latent frames measured 18.5/255 vs single-pass, 26.6% worst-frame clipping)",
            t.tile_frames
        );
        assert!(
            lat_over >= MIN_TEMPORAL_TILE_LATENT_OVERLAP,
            "{w}x{h}: temporal overlap {} output frames = {lat_over} LATENT frames after the tile-1 \
             clamp, under the {MIN_TEMPORAL_TILE_LATENT_OVERLAP}-frame blend floor",
            t.overlap_frames
        );
    }
    std::env::remove_var("WAN_VAE_BUDGET_GIB");
}

/// The request the real-weight smoke builds must be inside the model's *advertised* surface, checked
/// against the registered descriptor with no weights on disk. Discriminating: it goes through
/// `Generator::validate` (the same floor `run` self-applies), so a smoke that quietly drifted
/// out-of-surface — a guidance scale on this CFG-off model, an unadvertised sampler, a size below the
/// patch×stride alignment — fails here in CI instead of after a 20 GB load.
#[test]
fn smoke_request_is_within_the_advertised_surface() {
    fn tiny_safetensors(path: &Path, fill: u8) {
        let mut header = serde_json::to_vec(&serde_json::json!({
            "weight": {"dtype": "U8", "shape": [8], "data_offsets": [0, 8]}
        }))
        .unwrap();
        let padding = (8 - header.len() % 8) % 8;
        header.extend(std::iter::repeat_n(b' ', padding));
        let mut bytes = Vec::with_capacity(8 + header.len() + 8);
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&header);
        bytes.extend(std::iter::repeat_n(fill, 8));
        std::fs::write(path, bytes).unwrap();
    }
    // The provider now seals the complete physical receipt before construction, so this
    // weights-free validation uses a structurally valid tiny q4 inventory rather than an empty dir.
    let dir_tmp = tempfile::tempdir().unwrap();
    let dir = dir_tmp.path().to_path_buf();
    std::fs::write(
        dir.join("config.json"),
        br#"{"quantization":{"bits":4,"group_size":64}}"#,
    )
    .unwrap();
    std::fs::write(dir.join("tokenizer.json"), b"{}").unwrap();
    tiny_safetensors(&dir.join("dit.safetensors"), 1);
    tiny_safetensors(&dir.join("t5_encoder.safetensors"), 2);
    tiny_safetensors(&dir.join("vae.safetensors"), 3);
    let spec = LoadSpec::new(WeightsSource::Dir(dir));
    let gen = mlx_gen_krea_realtime::provider_registry()
        .unwrap()
        .load(MODEL_ID, &spec)
        .expect("load the krea_realtime_14b provider against an existing dir");
    assert_eq!(gen.descriptor().id, MODEL_ID);

    gen.validate(&smoke_request(832, 480, 81, None, 7))
        .expect("the t2v smoke request must validate");
    gen.validate(&i2v_request(832, 480, 81, None, 7, gradient(832, 480, 0)))
        .expect("the i2v smoke request must validate");
    let mut v2v = v2v_zero_request(832, 480, vec![gradient(832, 480, 0); 5]);
    gen.validate(&v2v)
        .expect("the strength-zero v2v smoke request must validate");
    v2v.video_mode = None;
    let err = gen
        .validate(&v2v)
        .expect_err("a VideoClip without an explicit v2v mode must fail");
    assert!(
        err.to_string().contains("video_mode=video_to_video"),
        "got: {err}"
    );

    // The floor really is being exercised: the same request plus a guidance scale is rejected on this
    // CFG-off model. Without this the check above could pass a validator that accepts everything.
    let mut cfg_on = smoke_request(832, 480, 81, None, 7);
    cfg_on.guidance = Some(5.0);
    let err = gen
        .validate(&cfg_on)
        .expect_err("a guidance scale must be rejected on a CFG-off model");
    assert!(err.to_string().contains("guidance"), "got: {err}");
}

// ---------------------------------------------------------------------------------------------
// Request builders (shared by the CI gate and the real-weight drivers)
// ---------------------------------------------------------------------------------------------

pub(super) fn smoke_request(
    w: usize,
    h: usize,
    frames: usize,
    steps: Option<usize>,
    seed: u64,
) -> GenerationRequest {
    GenerationRequest {
        prompt: "a red fox trotting through a snowy pine forest at sunrise, drifting snow, \
                 cinematic, shallow depth of field"
            .into(),
        width: w as u32,
        height: h as u32,
        frames: Some(frames as u32),
        steps: steps.map(|s| s as u32),
        seed: Some(seed),
        fps: Some(24),
        sampler: Some("self_forcing".into()),
        ..Default::default()
    }
}

pub(super) fn v2v_zero_request(w: usize, h: usize, source: Vec<Image>) -> GenerationRequest {
    let frames = source.len();
    GenerationRequest {
        video_mode: Some("video_to_video".into()),
        conditioning: vec![Conditioning::VideoClip {
            frames: source,
            frame_idx: 0,
            strength: 0.0,
        }],
        ..smoke_request(w, h, frames, None, 3)
    }
}

pub(super) fn i2v_request(
    w: usize,
    h: usize,
    frames: usize,
    steps: Option<usize>,
    seed: u64,
    reference: Image,
) -> GenerationRequest {
    GenerationRequest {
        conditioning: vec![Conditioning::Reference {
            image: reference,
            strength: None,
        }],
        ..smoke_request(w, h, frames, steps, seed)
    }
}

// ---------------------------------------------------------------------------------------------
// Real-weight drivers
// ---------------------------------------------------------------------------------------------

/// One instrumented `generate` run: returns the decoded clip plus the timing + memory measurements.
pub(super) struct RunResult {
    pub(super) frames: Vec<Image>,
    pub(super) fps: u32,
    pub(super) total: std::time::Duration,
    /// Wall time from the first `Progress::Step` to the last (the AR denoise phase only).
    pub(super) denoise: std::time::Duration,
    /// Per-chunk denoise wall times, derived from the step stream.
    pub(super) per_chunk: Vec<std::time::Duration>,
    pub(super) decode: std::time::Duration,
    /// Model load + prompt encode + the first denoise step (everything before the first `Step` mark).
    pub(super) prologue: std::time::Duration,
    /// Mean denoise-step wall time over **non-boundary** intervals — the cost of one denoise step, with
    /// the unobservable KV-recompute forward excluded. This is the figure epic 8432 should use.
    pub(super) mean_step: std::time::Duration,
    /// The same mean including boundary intervals (each of which also contains a KV recompute), kept so
    /// the inflation is visible rather than silently corrected away.
    pub(super) mean_step_with_boundaries: std::time::Duration,
    pub(super) peak_active: usize,
    pub(super) peak_total: usize,
    /// `(phase label, peak_active_so_far, peak_total_so_far)` at each pipeline boundary. Because both
    /// figures are monotonic, the phase at which they stop rising is the one that sets the ceiling.
    pub(super) phase_peaks: Vec<(&'static str, usize, usize)>,
}

pub(super) fn run(req: &GenerationRequest, expected_chunks: usize, label: &str) -> RunResult {
    let root = require_snapshot();
    let spec = LoadSpec::new(WeightsSource::Dir(root));
    let gen = mlx_gen_krea_realtime::provider_registry()
        .unwrap()
        .load(MODEL_ID, &spec)
        .expect("load the krea_realtime_14b provider");
    gen.validate(req).expect("validate");

    let sampler = MemorySampler::start();
    let t0 = Instant::now();
    let mut first_step: Option<Instant> = None;
    let mut step_marks: Vec<Instant> = Vec::new();
    let mut decode_start: Option<Instant> = None;
    let mut last = 0u32;
    let mut phase_peaks: Vec<(&'static str, usize, usize)> = Vec::new();

    let out = gen
        .generate(req, &mut |p| match p {
            Progress::Step { current, total } => {
                let now = Instant::now();
                assert!(current >= last, "{label}: progress went backwards");
                if first_step.is_none() {
                    first_step = Some(now);
                    println!("  {label}: {total} denoise steps");
                    let (a, t) = sampler.snapshot();
                    // NB: this fires at the FIRST `Progress::Step`, which is emitted *after* that
                    // step's compute and eval — so it already includes one full DiT forward and one
                    // chunk of KV, not just load + encode. Named for what it measures.
                    phase_peaks.push(("through the first denoise step", a, t));
                }
                last = current;
                step_marks.push(now);
            }
            Progress::Decoding => {
                decode_start = Some(Instant::now());
                let (a, t) = sampler.snapshot();
                phase_peaks.push(("after the AR denoise loop", a, t));
            }
            Progress::Loading(phase) => println!("  {label}: loading {phase:?}"),
            Progress::PromptUpsamplingComplete
            | Progress::PromptEncodingComplete
            | Progress::PostProcessing
            | Progress::ConditioningComplete
            | Progress::Upsampling { .. } => {}
        })
        .unwrap_or_else(|e| panic!("{label}: generate must succeed: {e}"));
    let total = t0.elapsed();
    let (peak_active, peak_total) = sampler.finish();
    phase_peaks.push(("after the VAE decode (final)", peak_active, peak_total));

    let GenerationOutput::Video { frames, fps, audio } = out else {
        panic!("{label}: expected a Video output");
    };
    assert!(audio.is_none(), "{label}: Krea Realtime has no audio track");

    // Per-chunk timing. The AR loop emits exactly one `Progress::Step` per denoise step and splits the
    // steps evenly across chunks, so chunk boundaries sit at `steps_per_chunk` strides through the
    // marks. Each mark is a step *completion*, so the interval between consecutive marks is one step.
    //
    // Chunk 0 is the one chunk whose start instant is not observable — the first mark already includes
    // the model load + prompt encode ahead of it. Its **first step** is therefore imputed at the mean
    // measured step time; chunks 1..N are exact mark-to-mark intervals. `prologue` (below) carries the
    // load + encode time separately so nothing is silently folded into a chunk number.
    let denoise = match (first_step, step_marks.last()) {
        (Some(a), Some(b)) => b.duration_since(a),
        _ => std::time::Duration::ZERO,
    };
    // Per-step time. An interval that SPANS a chunk boundary also contains the S5 clean-context
    // KV-recompute forward, which emits no `Progress` — including those inflates the per-step figure by
    // ~17% (measured 6.2 vs 5.3 s at 832x480). The realtime epic wants the cost of a denoise step, so
    // boundary intervals are excluded here and the recompute is reported inside the per-chunk time,
    // where it belongs.
    let all_intervals: Vec<std::time::Duration> = step_marks
        .windows(2)
        .map(|w| w[1].duration_since(w[0]))
        .collect();
    let per_chunk_steps = if expected_chunks > 0 && !step_marks.is_empty() {
        step_marks.len() / expected_chunks
    } else {
        0
    };
    let in_chunk: Vec<std::time::Duration> = all_intervals
        .iter()
        .enumerate()
        // interval i sits between step i and step i+1; it crosses a boundary when step i+1 starts a chunk
        .filter(|(i, _)| per_chunk_steps == 0 || !(i + 1).is_multiple_of(per_chunk_steps))
        .map(|(_, d)| *d)
        .collect();
    let mean_of = |v: &[std::time::Duration]| {
        if v.is_empty() {
            std::time::Duration::ZERO
        } else {
            v.iter().sum::<std::time::Duration>() / v.len() as u32
        }
    };
    let mean_step = mean_of(&in_chunk);
    let mean_step_with_boundaries = mean_of(&all_intervals);
    let mut per_chunk = Vec::new();
    if expected_chunks > 0
        && !step_marks.is_empty()
        && step_marks.len().is_multiple_of(expected_chunks)
    {
        let per = step_marks.len() / expected_chunks;
        for c in 0..expected_chunks {
            let end = step_marks[(c + 1) * per - 1];
            per_chunk.push(if c == 0 {
                end.duration_since(step_marks[0]) + mean_step
            } else {
                end.duration_since(step_marks[c * per - 1])
            });
        }
    }
    let prologue = first_step
        .map(|f| f.duration_since(t0))
        .unwrap_or(std::time::Duration::ZERO);
    let decode = decode_start
        .map(|d| (t0 + total).duration_since(d))
        .unwrap_or(std::time::Duration::ZERO);

    RunResult {
        frames,
        fps,
        total,
        denoise,
        per_chunk,
        decode,
        prologue,
        mean_step,
        mean_step_with_boundaries,
        peak_active,
        peak_total,
        phase_peaks,
    }
}

pub(super) fn report(label: &str, r: &RunResult, w: usize, h: usize, frames: usize) {
    println!("--- {label} -------------------------------------------------");
    println!(
        "  geometry     : {w}x{h}, {frames} frames @ {} fps ({} decoded)",
        r.fps,
        r.frames.len()
    );
    println!("  clip wall    : {:.2?}", r.total);
    println!(
        "  load+encode  : {:.2?} (to the first denoise-step mark)",
        r.prologue
    );
    println!(
        "  AR denoise   : {:.2?} ({:.2?}/step mean, excl. chunk boundaries; {:.2?} incl. — a \
         boundary interval also carries the KV-recompute forward, which emits no progress event)",
        r.denoise, r.mean_step, r.mean_step_with_boundaries
    );
    println!("  VAE decode   : {:.2?}", r.decode);
    for (i, d) in r.per_chunk.iter().enumerate() {
        println!(
            "    chunk {i:>2}   : {d:.2?}{}",
            if i == 0 {
                "  (first step imputed at the mean)"
            } else {
                ""
            }
        );
    }
    if !r.per_chunk.is_empty() {
        let mean = r.per_chunk.iter().sum::<std::time::Duration>() / r.per_chunk.len() as u32;
        println!("    chunk mean : {mean:.2?}");
    }
    println!(
        "  MLX peak     : active {:.2} GiB | active+cache {:.2} GiB  \
         (get_peak_memory is ACTIVE only — the cache is real resident memory too)",
        gib(r.peak_active),
        gib(r.peak_total)
    );
    for (phase, a, t) in &r.phase_peaks {
        println!(
            "    cumulative peak {phase:<28}: active {:.2} GiB | active+cache {:.2} GiB",
            gib(*a),
            gib(*t)
        );
    }
}
