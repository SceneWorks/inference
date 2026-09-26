//! **Real-weight device-memory measurement** (sc-19387, epic sc-19373): one full YuE render
//! through the registered loader at one of the SceneWorks request shapes the Metal memory capture
//! used, recording the device-memory peak overall and per engine phase plus wall time. It feeds the
//! SceneWorks CUDA admission estimate and the manifests' `candle` block; it asserts nothing about
//! the numbers (record-only).
//!
//! One case per process (`YUE_MEMORY_CASE` × `YUE_TIER`), so a case's peak never inherits a
//! previous case's allocator state. Cases — the Metal capture's shapes, same genre tags, lyrics and
//! seed as the SceneWorks Audio Studio requests (`sc-19387-t5-capture/make_cases.py`):
//!
//! | case          | model        | shape                                                                  |
//! |---------------|--------------|------------------------------------------------------------------------|
//! | `cot_default` | `yue_en_cot` | verse + chorus, engine defaults (2 segments × 3000 tokens, CFG)        |
//! | `icl_default` | `yue_en_icl` | same, dual reference 60 s, default window (0–30 s)                     |
//! | `cot_worst`   | `yue_en_cot` | 8 sections, 8 segments × 3000 tokens (stage-1 context at its 16 384 cap) |
//! | `icl_long`    | `yue_en_icl` | dual reference 217.68 s, window 0–120 s, 2 segments × 2000 tokens      |
//!
//! The dual references are upstream's `prompt_egs/pop.00001.{Vocals,Instrumental}.mp3` (30 s each,
//! 44.1 kHz stereo) decoded by `scripts/release/yue_real_weights.py stage-reference --with-stems`
//! and **looped** end-to-end to the capture's clip lengths (60 s and 217.68 s — the Metal capture's
//! long reference was a 217.68 s song). Only the window reaches the codec, so the clip body is
//! immaterial beyond its length. As in the SceneWorks worker, the pair travels as `vocals` +
//! `instrumental` stems of one `ReferenceAudio` whose mix is their sum.
//!
//! Phases, from the generator's progress events (the stage LMs report `Loading` only once loaded):
//! `registry_load` (the lazy load), `stage1_load` (ICL encode + prompt + stage-1 load),
//! `stage1_decode` (every segment), `stage2_load`, `stage2` (both tracks), `decode` (xcodec + Vocos
//! + splice).
//!
//! Memory, under `--features cuda` (CUDA ordinal 0, the device candle renders on):
//! - `device_*`: `cuMemGetInfo` used bytes sampled every 20 ms — the device-wide view NVML reports,
//!   so a co-tenant on the same GPU inflates it. `*_above_baseline` subtracts the GPU's
//!   `nvidia-smi` `memory.used` taken before this process created a CUDA context, so the context
//!   itself is charged to the render. On the Windows (WDDM) runner `cuMemGetInfo` also counts
//!   memory the driver reserves that nvidia-smi does not (~1.2 GiB over nvidia-smi's peak on the
//!   first capture); the workflow's own nvidia-smi series (split into these phases by `t0_unix`) is the
//!   NVML figure.
//! - `pool_*`: the stream-ordered memory pool candle allocates every tensor from (cudarc's
//!   `cuMemAllocAsync` on the device's current pool) — process-local, immune to co-tenants, and the
//!   pool's own high watermarks catch spikes between samples. `pool_reserved_high` is what the pool
//!   held from the driver; `pool_used_high` the live-tensor peak. cuBLAS workspaces and kernel
//!   modules sit outside the pool, which is the gap to `device_*`.
//!
//! Without `--features cuda` the render runs and only the timings are recorded.
//!
//! ```text
//! YUE_SNAPSHOT_ROOT=... YUE_REF_DIR=... YUE_MEMORY_CASE=cot_default YUE_TIER=q4 \
//!   YUE_MEMORY_OUT=<dir> cargo test --release -p candle-audio-yue --features cuda \
//!   --test cuda_memory_real_weights -- --ignored --exact measure_one_case --nocapture
//! ```
//!
//! Driven by `.github/workflows/real-weights-yue.yml` with `mode: memory`.

use std::path::{Path, PathBuf};
use std::time::Instant;

use candle_audio_yue::gen_core::{
    AudioParams, AudioStem, AudioTrack, Conditioning, GenerationOutput, GenerationRequest,
    LoadPhase, LoadSpec, Progress, Quant, TimeRegion, WeightsSource,
};
use candle_audio_yue::model::{SAMPLE_RATE, STAGE2_COMPONENT_ID, XCODEC_COMPONENT_ID};
use serde_json::{json, Value};

const PROMPT: &str = "inspiring female uplifting pop airy vocal electronic bright vocal";
const VERSE: &str = "Morning light is falling on the harbor wall\n\
Every gull is calling and I hear it all\n\
Paper boats are drifting where the river bends\n\
Carry every promise to the waiting friends";
const CHORUS: &str = "Hold on, hold on, the tide is turning home\n\
Sing it loud, sing it out, you are not alone\n\
Hold on, hold on, the lights are coming through\n\
Every road I wander brings me back to you";
const BRIDGE: &str = "Quiet in the evening when the lanterns glow\n\
Counting all the reasons that I never know";
const SEED: u64 = 42;

fn required_env(name: &str, what: &str) -> PathBuf {
    std::env::var_os(name)
        .unwrap_or_else(|| panic!("set {name} to {what}"))
        .into()
}

fn default_lyrics() -> String {
    format!("[verse]\n{VERSE}\n\n[chorus]\n{CHORUS}")
}

fn long_lyrics() -> String {
    [
        "verse", "chorus", "verse", "chorus", "bridge", "chorus", "verse", "chorus",
    ]
    .iter()
    .map(|s| {
        let body = match *s {
            "verse" => VERSE,
            "chorus" => CHORUS,
            _ => BRIDGE,
        };
        format!("[{s}]\n{body}")
    })
    .collect::<Vec<_>>()
    .join("\n\n")
}

/// `(tier directory, asserted quantize)` for `YUE_TIER`.
fn tier() -> (&'static str, Option<Quant>) {
    match std::env::var("YUE_TIER").as_deref() {
        Ok("q4") => ("q4", Some(Quant::Q4)),
        Ok("q8") => ("q8", Some(Quant::Q8)),
        Ok("bf16") => ("bf16", None),
        other => panic!("YUE_TIER must be q4, q8 or bf16, got {other:?}"),
    }
}

/// The worker's `LoadSpec` shape: each LM's selected tier directory, xcodec's repo root.
fn load_spec(stage1_repo: &str, tier_dir: &str, quantize: Option<Quant>) -> LoadSpec {
    let root = required_env(
        "YUE_SNAPSHOT_ROOT",
        "the directory holding the staged yue-s1-7b-anneal-*-candle, yue-s2-1b-general-candle and \
         xcodec-mini-infer snapshot roots",
    );
    let dir = |p: PathBuf| {
        assert!(p.is_dir(), "{} is not staged", p.display());
        WeightsSource::Dir(p)
    };
    let mut spec = LoadSpec::new(dir(root.join(stage1_repo).join(tier_dir)))
        .with_component(
            STAGE2_COMPONENT_ID,
            dir(root.join("yue-s2-1b-general-candle").join(tier_dir)),
        )
        .with_component(XCODEC_COMPONENT_ID, dir(root.join("xcodec-mini-infer")));
    spec.quantize = quantize;
    spec
}

/// One decoded upstream stem (`<clip>.f32le` + its `<clip>.json` rate/channels sidecar).
fn read_stem(dir: &Path, clip: &str) -> (Vec<f32>, u32, u16) {
    let meta: Value = serde_json::from_slice(
        &std::fs::read(dir.join(format!("{clip}.json")))
            .unwrap_or_else(|e| panic!("{clip}.json: {e} — stage it with --with-stems")),
    )
    .expect("stem sidecar is JSON");
    let raw = std::fs::read(dir.join(format!("{clip}.f32le")))
        .unwrap_or_else(|e| panic!("{clip}.f32le: {e}"));
    let samples = raw
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    (
        samples,
        meta["rate"].as_u64().expect("rate") as u32,
        meta["channels"].as_u64().expect("channels") as u16,
    )
}

/// The dual reference as the worker assembles it, each stem looped to `secs` seconds.
fn dual_reference(secs: f64) -> AudioTrack {
    let ref_dir = required_env(
        "YUE_REF_DIR",
        "the directory `yue_real_weights.py stage-reference --with-stems` populated",
    )
    .join("sceneworks-derived");
    let (vocals, rate, channels) = read_stem(&ref_dir, "pop.00001.Vocals");
    let (instrumental, i_rate, i_channels) = read_stem(&ref_dir, "pop.00001.Instrumental");
    assert_eq!(
        (rate, channels),
        (i_rate, i_channels),
        "stems differ in format"
    );
    let frames = (secs * f64::from(rate)).round() as usize;
    let len = frames * usize::from(channels);
    let looped = |s: &[f32]| -> Vec<f32> {
        // Whole frames only, so the channel interleave survives every wrap.
        let usable = s.len() - s.len() % usize::from(channels);
        assert!(usable > 0, "empty stem");
        s[..usable].iter().copied().cycle().take(len).collect()
    };
    let (vocals, instrumental) = (looped(&vocals), looped(&instrumental));
    let mix = vocals
        .iter()
        .zip(&instrumental)
        .map(|(v, i)| v + i)
        .collect();
    AudioTrack {
        samples: mix,
        sample_rate: rate,
        channels,
        stems: vec![
            AudioStem {
                name: "vocals".into(),
                samples: vocals,
            },
            AudioStem {
                name: "instrumental".into(),
                samples: instrumental,
            },
        ],
    }
}

/// `(model id, stage-1 repo, request, reference clip seconds)` for `YUE_MEMORY_CASE`.
fn case(name: &str) -> (&'static str, &'static str, GenerationRequest, Option<f64>) {
    let mut req = GenerationRequest {
        prompt: PROMPT.into(),
        seed: Some(SEED),
        audio: Some(AudioParams {
            lyrics: Some(default_lyrics()),
            ..Default::default()
        }),
        ..Default::default()
    };
    let audio = req.audio.as_mut().unwrap();
    let cot = ("yue_en_cot", "yue-s1-7b-anneal-en-cot-candle");
    let icl = ("yue_en_icl", "yue-s1-7b-anneal-en-icl-candle");
    let ((id, repo), clip) = match name {
        "cot_default" => (cot, None),
        "icl_default" => (icl, Some(60.0)),
        "cot_worst" => {
            audio.lyrics = Some(long_lyrics());
            audio.segments = Some(8);
            (cot, None)
        }
        "icl_long" => {
            audio.segments = Some(2);
            audio.max_new_tokens_per_segment = Some(2000);
            audio.reference_region = Some(TimeRegion {
                start_secs: 0.0,
                end_secs: Some(120.0),
            });
            (icl, Some(217.68))
        }
        other => panic!(
            "YUE_MEMORY_CASE must be cot_default, icl_default, cot_worst or icl_long, got {other}"
        ),
    };
    if let Some(secs) = clip {
        req.conditioning = vec![Conditioning::ReferenceAudio {
            audio: dual_reference(secs),
            strength: None,
        }];
    }
    (id, repo, req, clip)
}

#[cfg(feature = "cuda")]
mod probe {
    //! CUDA memory probes: `cuMemGetInfo` (device-wide) and the device's current stream-ordered
    //! memory pool (the one candle allocates from through cudarc's `cuMemAllocAsync`).
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};

    use candle_audio_yue::candle_audio::candle_core::cuda_backend::cudarc::driver::{
        result, sys, CudaContext,
    };

    use sys::CUmemPool_attribute as A;

    /// PCI bus id of CUDA ordinal 0 in `nvidia-smi -i` form, read WITHOUT creating a context so the
    /// baseline taken with it excludes this process.
    pub fn pci_bus_id() -> String {
        result::init().expect("cuInit");
        let dev = result::device::get(0).expect("CUDA ordinal 0");
        let attr = |a| unsafe { result::device::get_attribute(dev, a) }.expect("device attribute");
        use sys::CUdevice_attribute as D;
        format!(
            "{:08X}:{:02X}:{:02X}.0",
            attr(D::CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID),
            attr(D::CU_DEVICE_ATTRIBUTE_PCI_BUS_ID),
            attr(D::CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID)
        )
    }

    /// One sample: seconds since the probe started, device used, pool used, pool reserved.
    pub type Sample = (f64, u64, u64, u64);

    pub struct Probe {
        ctx: Arc<CudaContext>,
        // `CUmemoryPool` is a raw pointer (not `Send`); the handle is process-global.
        pool: usize,
        start: Instant,
        stop: Arc<AtomicBool>,
        samples: Arc<Mutex<Vec<Sample>>>,
        thread: Option<std::thread::JoinHandle<()>>,
    }

    fn pool_attr(pool: usize, attr: A) -> u64 {
        let mut v: u64 = 0;
        unsafe {
            result::mem_pool::get_attribute(
                pool as sys::CUmemoryPool,
                attr,
                (&mut v as *mut u64).cast(),
            )
        }
        .expect("cuMemPoolGetAttribute");
        v
    }

    fn device_used(ctx: &CudaContext) -> u64 {
        ctx.bind_to_thread().expect("bind context");
        let (free, total) = result::mem_get_info().expect("cuMemGetInfo");
        (total - free) as u64
    }

    impl Probe {
        /// Retain the primary context of ordinal 0 (the one candle uses) and start sampling.
        pub fn start() -> Self {
            let ctx = CudaContext::new(0).expect("CUDA context");
            let pool = unsafe { result::device::get_mem_pool(ctx.cu_device()) }
                .expect("cuDeviceGetMemPool") as usize;
            let start = Instant::now();
            let stop = Arc::new(AtomicBool::new(false));
            let samples = Arc::new(Mutex::new(Vec::new()));
            let thread = {
                let (ctx, stop, samples) = (ctx.clone(), stop.clone(), samples.clone());
                std::thread::spawn(move || {
                    while !stop.load(Ordering::Relaxed) {
                        let s = (
                            start.elapsed().as_secs_f64(),
                            device_used(&ctx),
                            pool_attr(pool, A::CU_MEMPOOL_ATTR_USED_MEM_CURRENT),
                            pool_attr(pool, A::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT),
                        );
                        samples.lock().unwrap().push(s);
                        std::thread::sleep(Duration::from_millis(20));
                    }
                })
            };
            let probe = Self {
                ctx,
                pool,
                start,
                stop,
                samples,
                thread: Some(thread),
            };
            probe.take_highs();
            probe
        }

        pub fn device_used_now(&self) -> u64 {
            device_used(&self.ctx)
        }

        /// The pool's `(used, reserved)` high watermarks since the last call, then reset them.
        pub fn take_highs(&self) -> (u64, u64) {
            let highs = (
                pool_attr(self.pool, A::CU_MEMPOOL_ATTR_USED_MEM_HIGH),
                pool_attr(self.pool, A::CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH),
            );
            for attr in [
                A::CU_MEMPOOL_ATTR_USED_MEM_HIGH,
                A::CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH,
            ] {
                let mut zero: u64 = 0;
                unsafe {
                    result::mem_pool::set_attribute(
                        self.pool as sys::CUmemoryPool,
                        attr,
                        (&mut zero as *mut u64).cast(),
                    )
                }
                .expect("reset pool high watermark");
            }
            highs
        }

        /// The sampler's clock origin; phase marks use it so samples and marks share a timebase.
        pub fn origin(&self) -> Instant {
            self.start
        }

        pub fn finish(mut self) -> Vec<Sample> {
            self.stop.store(true, Ordering::Relaxed);
            if let Some(t) = self.thread.take() {
                t.join().expect("sampler thread");
            }
            std::mem::take(&mut *self.samples.lock().unwrap())
        }
    }
}

/// `nvidia-smi` `memory.used` (MiB) of the GPU at `pci`, or `None` without nvidia-smi.
#[cfg(feature = "cuda")]
fn nvidia_smi_used_mib(pci: &str) -> Option<u64> {
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
            "-i",
            pci,
        ])
        .output()
        .ok()?;
    String::from_utf8_lossy(&out.stdout).trim().parse().ok()
}

#[cfg(feature = "cuda")]
const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

#[test]
#[ignore = "real weights: set YUE_SNAPSHOT_ROOT, YUE_REF_DIR, YUE_MEMORY_CASE, YUE_TIER (module docs)"]
fn measure_one_case() {
    let case_name = std::env::var("YUE_MEMORY_CASE").expect("set YUE_MEMORY_CASE");
    let (tier_dir, quantize) = tier();
    let out_dir = required_env(
        "YUE_MEMORY_OUT",
        "the directory the case JSON is written to",
    );
    let (id, repo, req, clip_secs) = case(&case_name);
    let spec = load_spec(repo, tier_dir, quantize);

    #[cfg(feature = "cuda")]
    let (pci, baseline_mib) = {
        let pci = probe::pci_bus_id();
        (pci.clone(), nvidia_smi_used_mib(&pci))
    };
    #[cfg(feature = "cuda")]
    let probe = probe::Probe::start();
    #[cfg(feature = "cuda")]
    let context_used = probe.device_used_now();

    // Phase marks: (label of the phase that just ENDED, seconds since t0, pool highs inside it).
    #[cfg(feature = "cuda")]
    let t0 = probe.origin();
    #[cfg(not(feature = "cuda"))]
    let t0 = Instant::now();
    // Wall-clock anchor of t0, so an external sampler's timestamps can be split into the phases.
    let t0_unix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock after the epoch")
        .as_secs_f64()
        - t0.elapsed().as_secs_f64();
    let mut marks: Vec<(&'static str, f64, (u64, u64))> = Vec::new();
    let mark = |label: &'static str, marks: &mut Vec<_>| {
        #[cfg(feature = "cuda")]
        let highs = probe.take_highs();
        #[cfg(not(feature = "cuda"))]
        let highs = (0u64, 0u64);
        marks.push((label, t0.elapsed().as_secs_f64(), highs));
    };

    let registry = candle_audio_yue::provider_registry().expect("registry builds");
    let generator = registry
        .load(id, &spec)
        .unwrap_or_else(|e| panic!("{id}: the registered loader refused: {e}"));
    generator.validate(&req).expect("the request is valid");
    mark("registry_load", &mut marks);

    let mut loads = 0;
    let mut progress = Vec::new();
    let out = generator
        .generate(&req, &mut |p| {
            let at = t0.elapsed().as_secs_f64();
            match &p {
                Progress::Loading(LoadPhase::Renderer) => {
                    loads += 1;
                    mark(
                        if loads == 1 {
                            "stage1_load"
                        } else {
                            "stage2_load"
                        },
                        &mut marks,
                    );
                }
                Progress::Step { current, total } if *current + 2 == *total => {
                    mark("stage1_decode", &mut marks)
                }
                Progress::Decoding => mark("stage2", &mut marks),
                _ => {}
            }
            progress.push(format!("{at:.2} {p:?}"));
        })
        .unwrap_or_else(|e| panic!("{id}: render failed: {e}"));
    mark("decode", &mut marks);
    let wall = t0.elapsed().as_secs_f64();

    let GenerationOutput::Audio(track) = out else {
        panic!("{id}: expected audio output");
    };
    assert_eq!(track.sample_rate, SAMPLE_RATE);
    assert!(
        track.samples.iter().all(|x| x.is_finite()),
        "non-finite mix"
    );
    let song_secs = track.samples.len() as f64 / f64::from(track.sample_rate);

    #[cfg(feature = "cuda")]
    let samples = probe.finish();
    #[cfg(feature = "cuda")]
    let (base_bytes, device_peak) = (
        baseline_mib.map(|m| m * 1024 * 1024),
        samples.iter().map(|s| s.1).max().unwrap_or(0),
    );

    let mut phases = Vec::new();
    let mut start = 0.0;
    for (label, end, (pool_used_high, pool_reserved_high)) in &marks {
        #[allow(unused_mut)]
        let mut phase = json!({
            "phase": label,
            "start_s": start,
            "end_s": end,
            "wall_s": end - start,
        });
        #[cfg(feature = "cuda")]
        {
            let inside: Vec<_> = samples
                .iter()
                .filter(|s| s.0 >= start && s.0 <= *end)
                .collect();
            let dev = inside.iter().map(|s| s.1).max();
            phase["device_used_peak_bytes"] = json!(dev);
            phase["device_peak_above_baseline_gib"] = json!(dev
                .zip(base_bytes)
                .map(|(d, b)| (d.saturating_sub(b)) as f64 / GIB));
            phase["pool_used_high_bytes"] = json!(pool_used_high);
            phase["pool_reserved_high_bytes"] = json!(pool_reserved_high);
            phase["pool_used_high_gib"] = json!(*pool_used_high as f64 / GIB);
            phase["pool_reserved_high_gib"] = json!(*pool_reserved_high as f64 / GIB);
        }
        #[cfg(not(feature = "cuda"))]
        let _ = (pool_used_high, pool_reserved_high);
        phases.push(phase);
        start = *end;
    }

    #[allow(unused_mut)]
    let mut record = json!({
        "case": case_name,
        "tier": tier_dir,
        "model": id,
        "request": {
            "segments": req.audio.as_ref().and_then(|a| a.segments),
            "max_new_tokens_per_segment": req.audio.as_ref().and_then(|a| a.max_new_tokens_per_segment),
            "reference_clip_secs": clip_secs,
            "reference_region": req.audio.as_ref().and_then(|a| a.reference_region).map(|r| (r.start_secs, r.end_secs)),
            "seed": SEED,
        },
        "t0_unix": t0_unix,
        "wall_s": wall,
        "song_s": song_secs,
        "phases": phases,
        "progress": progress,
    });
    #[cfg(feature = "cuda")]
    {
        let (used_high_all, reserved_high_all) = marks
            .iter()
            .fold((0, 0), |acc, m| (acc.0.max(m.2 .0), acc.1.max(m.2 .1)));
        record["cuda"] = json!({
            "ordinal": 0,
            "pci_bus_id": pci,
            "baseline_nvidia_smi_mib": baseline_mib,
            "device_used_after_context_bytes": context_used,
            "device_used_peak_bytes": device_peak,
            "device_peak_above_baseline_gib":
                base_bytes.map(|b| device_peak.saturating_sub(b) as f64 / GIB),
            "pool_used_high_gib": used_high_all as f64 / GIB,
            "pool_reserved_high_gib": reserved_high_all as f64 / GIB,
            "samples": samples.len(),
            // Per-second maxima of the 20 ms series: (s, device, pool used, pool reserved) bytes.
            "timeline_1s_max": per_second_max(&samples),
        });
    }

    std::fs::create_dir_all(&out_dir).expect("create YUE_MEMORY_OUT");
    let path = out_dir.join(format!("{case_name}-{tier_dir}.json"));
    std::fs::write(&path, serde_json::to_vec_pretty(&record).unwrap()).expect("write case JSON");
    println!(
        "{case_name} {tier_dir}: {song_secs:.1} s song in {wall:.1} s; {}",
        record
            .get("cuda")
            .map(|c| format!(
                "device peak above baseline {:.2} GiB, pool reserved high {:.2} GiB, pool used high {:.2} GiB",
                c["device_peak_above_baseline_gib"].as_f64().unwrap_or(f64::NAN),
                c["pool_reserved_high_gib"].as_f64().unwrap_or(f64::NAN),
                c["pool_used_high_gib"].as_f64().unwrap_or(f64::NAN),
            ))
            .unwrap_or_else(|| "no CUDA probe (built without --features cuda)".into())
    );
    println!("wrote {}", path.display());
}

#[cfg(feature = "cuda")]
fn per_second_max(samples: &[probe::Sample]) -> Vec<(u64, u64, u64, u64)> {
    let mut out: Vec<(u64, u64, u64, u64)> = Vec::new();
    for &(t, d, u, r) in samples {
        let sec = t as u64;
        match out.last_mut() {
            Some(last) if last.0 == sec => {
                last.1 = last.1.max(d);
                last.2 = last.2.max(u);
                last.3 = last.3.max(r);
            }
            _ => out.push((sec, d, u, r)),
        }
    }
    out
}
