//! Runtime diagnostic: does the fatbin this binary was built with actually cover the GPU it is
//! running on? (sc-19545)
//!
//! # The failure this explains
//!
//! `candle-kernels` compiles its GGUF `QMatMul` kernels (`mmq_gguf/*`, `moe/*`, `mmvq_gguf`) with
//! `nvcc -c` — a SASS object with **no PTX**. If none of the compiled architectures serves the GPU
//! actually present, the launch does not error. It writes nothing. Dense models still render,
//! quantized models come out **black**, and the process exits 0 with nothing in the log to explain
//! it. That is sc-7544, and it is the worst failure shape available: silent, wrong, unattributable.
//!
//! This module makes it attributable. At CUDA device construction it compares the device's real
//! compute capability against the ladder baked in at build time and, on no match, prints a loud
//! warning naming the capability, the rungs actually present, and the consequence.
//!
//! # Deliberately NON-FATAL, pending verification on Blackwell hardware
//!
//! It warns; it never returns `Err`, panics, or aborts a render. The asymmetry is the whole
//! argument: a false positive costs one spurious log line and breaks nothing, while a correct fire
//! finally explains a black render at the moment it happens. A hard failure would invert that —
//! a false positive would break renders that work today, and **no one has yet executed this check
//! on a Blackwell (sm_120) box** to establish that it does not misfire. Upgrading it to a hard
//! error is a considered next step with a named prerequisite (one clean run on that hardware), not
//! a loose end.
//!
//! # Where the ladder comes from
//!
//! `build.rs` parses `vendor/candle-kernels/build.rs` — the file that actually passes the
//! `-gencode` flags — and bakes the result in as env vars. Nothing here restates the ladder, so it
//! cannot claim coverage the build does not have.
//!
//! # Compute capabilities are packed integers
//!
//! `80` is sm_8.0, `120` is sm_12.0: `major = cap / 10`, `minor = cap % 10`. CUDA's own
//! `CUDA_COMPUTE_CAP` / `-gencode` spelling, so no conversion is needed anywhere.

// Only the cuda-gated warn path uses this; an unconditional import is an unused-import error on the
// CPU/Metal lanes, which build this module for its pure comparison logic and unit tests.
#[cfg(feature = "cuda")]
use std::sync::Once;

/// The architectures a build can actually execute on, split by compile path.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FatbinLadder {
    /// Native SASS cubins in `libmoe.a` — the cudaforge baseline rung plus the vendored fork's
    /// explicit `code=sm_NN` flags.
    pub sass: Vec<u32>,
    /// Embedded PTX floors (`code=compute_NN`). PTX JITs forward, so a floor covers every arch at
    /// or above it.
    pub ptx: Vec<u32>,
}

impl FatbinLadder {
    /// The ladder baked in by `build.rs`. Empty when the vendored source could not be read, which
    /// makes [`describe_if_uncovered`] stay silent rather than guess.
    pub fn from_build() -> Self {
        Self::parse(
            env!("CANDLE_GEN_FATBIN_SASS"),
            env!("CANDLE_GEN_FATBIN_PTX"),
            env!("CANDLE_GEN_FATBIN_BASELINE"),
        )
    }

    /// Assemble a ladder from the three build-script strings.
    ///
    /// Separate from [`from_build`] because those are `env!` values fixed at compile time, so the
    /// assembly — in particular folding in the baseline rung — is untestable through that door. It
    /// is not a detail: the baseline is where sm_80 comes from, and losing it silently narrows the
    /// ladder to the fork's explicit rungs.
    ///
    /// **A missing baseline is treated as unknown, not as absent.** A `--features cuda` build with
    /// no `CUDA_COMPUTE_CAP` set lets cudaforge auto-detect the cap off the build host, so a rung
    /// exists that this process cannot name. Reporting the ladder without it could warn about a
    /// device that is in fact served. All 15 CI sites set the variable, so this is the local
    /// hand-build case; it resolves to silence rather than a false alarm.
    fn parse(sass_csv: &str, ptx_csv: &str, baseline_csv: &str) -> Self {
        fn csv(raw: &str) -> Vec<u32> {
            raw.split(',')
                .filter_map(|v| v.trim().parse().ok())
                .collect()
        }
        let mut sass = csv(sass_csv);
        let ptx = csv(ptx_csv);
        match baseline_csv.trim().parse::<u32>() {
            // The cudaforge `-gencode` that `CUDA_COMPUTE_CAP` itself contributes. Not one of the
            // fork's explicit flags, so it has to be folded in here.
            Ok(baseline) => sass.push(baseline),
            // Unknown baseline: degrade to an empty ladder, which never warns.
            Err(_) if !sass.is_empty() => return Self::default(),
            Err(_) => {}
        }
        sass.sort_unstable();
        sass.dedup();
        Self { sass, ptx }
    }

    /// Whether the quantized kernels can run on `device_cap`.
    ///
    /// Two rules, both CUDA's rather than ours:
    ///
    /// * **SASS** is binary-compatible upward *within one major version only*. An sm_80 cubin runs
    ///   on sm_86 and sm_89; it never runs on sm_90 or sm_120.
    /// * **PTX** JITs forward across majors, so a `compute_N` floor covers any arch `>= N`.
    ///
    /// An empty ladder returns `true` — "unknown" must not be reported as "broken".
    pub fn covers(&self, device_cap: u32) -> bool {
        if self.sass.is_empty() && self.ptx.is_empty() {
            return true;
        }
        self.sass
            .iter()
            .any(|&c| c / 10 == device_cap / 10 && c % 10 <= device_cap % 10)
            || self.ptx.iter().any(|&floor| floor <= device_cap)
    }
}

/// The warning text for an uncovered device, or `None` when the device is served.
///
/// Split out from the printing so the decision is a pure function: the unit tests below drive it
/// with synthetic ladders, which needs no GPU and no CUDA build.
pub fn describe_if_uncovered(ladder: &FatbinLadder, device_cap: u32) -> Option<String> {
    if ladder.covers(device_cap) {
        return None;
    }
    let (major, minor) = (device_cap / 10, device_cap % 10);
    let rungs = if ladder.sass.is_empty() {
        "none".to_string()
    } else {
        ladder
            .sass
            .iter()
            .map(|c| format!("sm_{c}"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    let floors = if ladder.ptx.is_empty() {
        "none".to_string()
    } else {
        ladder
            .ptx
            .iter()
            .map(|c| format!("compute_{c}"))
            .collect::<Vec<_>>()
            .join(", ")
    };
    // Datacenter Blackwell is the known, measured instance of this gap, so name it rather than
    // leaving the reader to work out why a Blackwell card is uncovered on a Blackwell-aware build.
    // sm_100 is major 10: the sm_12x cubins do not serve it (different major) and a compute_120
    // floor is above it. `vendor/candle-kernels/build.rs` says this is deliberate.
    let known = match (major, minor) {
        (10, 0) => {
            "\n  This is the known B100/B200 (datacenter Blackwell) gap: sm_100 is major 10, so \
             the sm_12x cubins do not serve it and the compute_120 PTX floor is above it. \
             vendor/candle-kernels/build.rs targets it deliberately out of scope — add \
             `-gencode=arch=compute_100,code=sm_100` there if this card is now supported."
        }
        (10, _) => {
            "\n  Major 10 (datacenter Blackwell, B100/B200 family) is deliberately out of scope in \
             vendor/candle-kernels/build.rs."
        }
        _ => "",
    };
    Some(format!(
        "\n\
         ============================ candle-gen: CUDA ARCH WARNING ============================\n\
         This GPU is sm_{device_cap} (compute capability {major}.{minor}), which NO architecture in \
         this build's\n\
         quantized-kernel fatbin serves.\n\
         \n  native SASS cubins: {rungs}\n  embedded PTX floors: {floors}\n\
         \n\
         CONSEQUENCE: quantized (Q4/Q8 GGUF) matmuls will SILENTLY RETURN ZEROS on this device. \
         They\n\
         do not error — RENDERS MAY PRODUCE BLACK OUTPUT while every exit code stays 0. Dense \
         (non-\n\
         quantized) models are unaffected; they ship as forward-JITable PTX.\n\
         \n\
         FIX: add this architecture to the -gencode ladder in \
         crates/media/candle-gen/vendor/candle-kernels/build.rs.\n\
         Do NOT instead raise CUDA_COMPUTE_CAP to {device_cap}: that variable is the ladder's \
         BOTTOM rung, and\n\
         raising it deletes the oldest rung, breaking every older GPU.{known}\n\
         =======================================================================================\n"
    ))
}

/// Emit the warning at most once per process. Called from `default_device()` on the CUDA path.
///
/// Gated with its only caller: `default_device()` may be called per model load, so the `Once` is
/// what keeps a real fire from becoming log spam — but on CPU/Metal builds there is no caller and
/// an ungated copy is dead code under `-D warnings`.
#[cfg(feature = "cuda")]
fn warn_once(message: &str) {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| eprintln!("{message}"));
}

/// Check the CUDA device against the baked-in ladder and warn if it is not covered.
///
/// Best-effort throughout: a device whose capability cannot be read is left alone. Never fails.
#[cfg(feature = "cuda")]
pub fn warn_if_device_uncovered(device: &candle_core::Device) {
    use candle_core::cuda::cudarc::driver::sys::CUdevice_attribute as Attr;
    let candle_core::Device::Cuda(cuda) = device else {
        return;
    };
    let stream = cuda.cuda_stream();
    let ctx = stream.context();
    let (Ok(major), Ok(minor)) = (
        ctx.attribute(Attr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR),
        ctx.attribute(Attr::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR),
    ) else {
        return;
    };
    let cap = major as u32 * 10 + minor as u32;
    if let Some(message) = describe_if_uncovered(&FatbinLadder::from_build(), cap) {
        warn_once(&message);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shipped ladder as of sc-7544: sm_80 baseline + sm_90 + sm_120 SASS, compute_120 PTX.
    fn shipped() -> FatbinLadder {
        FatbinLadder {
            sass: vec![80, 90, 120],
            ptx: vec![120],
        }
    }

    #[test]
    fn the_shipped_ladder_covers_every_architecture_it_claims() {
        let ladder = shipped();
        for cap in [80, 86, 89, 90, 120, 121, 130] {
            assert!(ladder.covers(cap), "sm_{cap} should be covered");
            assert!(describe_if_uncovered(&ladder, cap).is_none());
        }
    }

    /// SASS compatibility is upward WITHIN one major and no further. This is the rule that decides
    /// sm_100, so state it on its own rather than only via that case.
    #[test]
    fn sass_does_not_cross_a_major_version() {
        let ampere_only = FatbinLadder {
            sass: vec![80],
            ptx: vec![],
        };
        assert!(ampere_only.covers(86), "sm_80 cubin serves sm_86");
        assert!(ampere_only.covers(89), "sm_80 cubin serves sm_89");
        assert!(!ampere_only.covers(90), "sm_80 cubin must NOT serve sm_90");
        assert!(
            !ampere_only.covers(120),
            "sm_80 cubin must NOT serve sm_120"
        );
        // ...and not downward within the major either.
        assert!(!FatbinLadder {
            sass: vec![86],
            ptx: vec![]
        }
        .covers(80));
    }

    #[test]
    fn ptx_floors_jit_forward_across_majors_but_never_backward() {
        let ladder = FatbinLadder {
            sass: vec![],
            ptx: vec![90],
        };
        assert!(ladder.covers(90));
        assert!(ladder.covers(120), "compute_90 PTX JITs forward to sm_120");
        assert!(!ladder.covers(80), "sm_80 is below the compute_90 floor");
    }

    /// The measured finding this diagnostic was written for: datacenter Blackwell is served by no
    /// rung of the shipped ladder, and the warning must say so by name.
    #[test]
    fn datacenter_blackwell_sm_100_is_uncovered_and_named() {
        let ladder = shipped();
        assert!(!ladder.covers(100));
        let message = describe_if_uncovered(&ladder, 100).expect("sm_100 must warn");
        assert!(message.contains("sm_100"));
        assert!(message.contains("B100/B200"));
        assert!(message.contains("SILENTLY RETURN ZEROS"));
        assert!(message.contains("BLACK OUTPUT"));
        // It must not send the reader down the road this whole story exists to close off.
        assert!(message.contains("Do NOT instead raise CUDA_COMPUTE_CAP"));
    }

    #[test]
    fn turing_is_uncovered_by_the_shipped_ladder() {
        let ladder = shipped();
        assert!(!ladder.covers(75));
        let message = describe_if_uncovered(&ladder, 75).expect("sm_75 must warn");
        assert!(message.contains("sm_75"));
        // Not the datacenter-Blackwell note — that is major 10 only.
        assert!(!message.contains("B100/B200"));
    }

    /// An unreadable vendor source yields an empty ladder, and "unknown" must never be reported as
    /// "broken" — a diagnostic that fires on every CPU build would be deleted within a week.
    #[test]
    fn an_empty_ladder_never_warns() {
        let empty = FatbinLadder::default();
        for cap in [75, 80, 100, 120] {
            assert!(empty.covers(cap));
            assert!(describe_if_uncovered(&empty, cap).is_none());
        }
    }

    /// The baseline rung is where sm_80 comes from. Dropping it narrows the ladder to the fork's
    /// explicit rungs and would make the diagnostic warn on every Ampere card — a mutation that no
    /// literal-constructed ladder above can see, because none of them goes through `parse`.
    #[test]
    fn parse_folds_the_baseline_rung_into_the_sass_ladder() {
        let ladder = FatbinLadder::parse("90,120", "120", "80");
        assert_eq!(
            ladder.sass,
            vec![80, 90, 120],
            "baseline 80 must be folded in"
        );
        assert_eq!(ladder.ptx, vec![120]);
        assert!(
            ladder.covers(86),
            "Ampere is served only via the baseline rung"
        );

        // Deduplicated and ordered, so a baseline that repeats an explicit rung is harmless.
        assert_eq!(FatbinLadder::parse("120,90", "", "90").sass, vec![90, 120]);
    }

    /// An unknown baseline must not be reported as a NARROWER ladder — that would warn about
    /// devices the build can in fact serve. Silence is the correct degradation.
    #[test]
    fn parse_degrades_to_silence_when_the_baseline_is_unknown() {
        let ladder = FatbinLadder::parse("90,120", "120", "");
        assert_eq!(ladder, FatbinLadder::default());
        assert!(
            ladder.covers(80),
            "unknown must never be reported as broken"
        );
        assert!(describe_if_uncovered(&ladder, 80).is_none());
    }

    /// A vendor source that could not be read yields empty strings throughout, which is the other
    /// route to the silent ladder.
    #[test]
    fn parse_of_an_unreadable_vendor_source_is_empty() {
        assert_eq!(FatbinLadder::parse("", "", ""), FatbinLadder::default());
    }

    /// Mutation coverage for the comparison logic: each way the predicate could be wrong produces a
    /// ladder/device pair whose verdict flips. If any assertion here can be deleted without a test
    /// going red, the predicate is not pinned.
    #[test]
    fn the_coverage_predicate_discriminates_mutations() {
        let ladder = shipped();

        // Mutation: drop the Blackwell SASS rung. The compute_120 PTX floor still serves sm_120, so
        // the verdict must NOT change — a predicate that ignored PTX would flip here and start
        // crying wolf on the actual production runner.
        let no_sm120 = FatbinLadder {
            sass: vec![80, 90],
            ptx: vec![120],
        };
        assert!(no_sm120.covers(120));

        // Mutation: drop the PTX floor as well. NOW sm_120 has neither a cubin nor anything to JIT.
        let no_blackwell = FatbinLadder {
            sass: vec![80, 90],
            ptx: vec![],
        };
        assert!(!no_blackwell.covers(120));

        // Mutation: revert sc-7544 entirely (baseline rung only). Only Ampere survives.
        let ampere_only = FatbinLadder {
            sass: vec![80],
            ptx: vec![],
        };
        assert!(ampere_only.covers(80));
        assert!(!ampere_only.covers(90));
        assert!(!ampere_only.covers(120));

        // Mutation: raise CUDA_COMPUTE_CAP to the runner's own arch, which drops the bottom rung.
        // Every pre-Blackwell GPU loses SASS coverage. This is the change sc-19545 was filed to
        // request, and the reason it was refused.
        let cap_raised_to_120 = FatbinLadder {
            sass: vec![90, 120, 120],
            ptx: vec![120],
        };
        assert!(!cap_raised_to_120.covers(80));
        assert!(!cap_raised_to_120.covers(86));
        assert!(ladder.covers(86), "the shipped ladder does serve sm_86");

        // Mutation: a minor-version-only comparison (forgetting the major) would call sm_100
        // covered by sm_120, since both have minor 0.
        assert!(!ladder.covers(100));
        // Mutation: a `>=` on the whole packed integer would call sm_100 covered by sm_90.
        assert!(!FatbinLadder {
            sass: vec![90],
            ptx: vec![]
        }
        .covers(100));
    }
}
