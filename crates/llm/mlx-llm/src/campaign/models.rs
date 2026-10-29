use super::*;

/// One file which must be present, byte-for-byte, in a SC-20671 benchmark snapshot.
/// Binding every published safetensors payload and the parsed config is sufficient to reject a
/// caller-provided model substitution without inventing a cache path convention.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PinnedSnapshotFile {
    pub path: &'static str,
    pub bytes: u64,
    pub sha256: &'static str,
}

/// Source-owned identity for one of the four dense-baseline snapshots.  These are commit IDs, not
/// mutable Hugging Face branches or aliases.  Snapshot directories intentionally remain CLI
/// inputs: the operator controls storage, while this contract controls what may be loaded from it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BenchmarkModelSpec {
    pub family: &'static str,
    pub role: &'static str,
    pub repository: &'static str,
    pub revision: &'static str,
    pub architecture: &'static str,
    pub model_type: &'static str,
    pub native_context_tokens: u64,
    /// The fit-boundary row's window when it is not the native window: the largest window whose
    /// fit row the 68 GiB child footprint cap of `.github/kv-poc/policies/llm.json` admits with
    /// margin (see [`spec_band_target`]). `None` = native.
    pub fit_window_tokens: Option<u64>,
    pub quantized: bool,
    pub required_files: &'static [PinnedSnapshotFile],
}

pub(super) const LLAMA_4BIT_FILES: &[PinnedSnapshotFile] = &[
    PinnedSnapshotFile {
        path: "config.json",
        bytes: 1122,
        sha256: "c546925585e48f43890d9dc5150df4fec73dd3780d92961c5ace451934cc4cd6",
    },
    PinnedSnapshotFile {
        path: "model.safetensors",
        bytes: 1_807_496_278,
        sha256: "d75e1ee0ea653cc5b76191ec934c7c0d568e94d4e47846619f1f4bc715b7b265",
    },
    PinnedSnapshotFile {
        path: "model.safetensors.index.json",
        bytes: 45_720,
        sha256: "2ef31fa0b9dcda01f87835851d5e1d5a39ab6258ae618af6ea864c3747e556e4",
    },
    PinnedSnapshotFile {
        path: "tokenizer_config.json",
        bytes: 54_558,
        sha256: "022d5ae3df4737998ab97d8f31ac2bcb4c06dd8ebe5a8aba2b4aceef1e5ea7d3",
    },
];

pub(super) const LLAMA_BF16_FILES: &[PinnedSnapshotFile] = &[
    PinnedSnapshotFile {
        path: "config.json",
        bytes: 969,
        sha256: "7e4149635018dd8d82f9b0873d800459faca261b1ac969d458c93f51133643a8",
    },
    PinnedSnapshotFile {
        path: "model-00001-of-00002.safetensors",
        bytes: 5_368_478_882,
        sha256: "b475af8535933afaf9393b30c6bcb4bfede59586b16ecdd010e7da8a83dfc95b",
    },
    PinnedSnapshotFile {
        path: "model-00002-of-00002.safetensors",
        bytes: 1_057_050_089,
        sha256: "9b2903fa2b3728423f6442c0dc667db645906fb113c9d0cd794faeb820cc82d3",
    },
    PinnedSnapshotFile {
        path: "model.safetensors.index.json",
        bytes: 21_946,
        sha256: "de5995e586fc98a942b576a6752034a6d0a741d2dece5f61e697d99c8bb99b3d",
    },
    PinnedSnapshotFile {
        path: "tokenizer_config.json",
        bytes: 54_528,
        sha256: "9823dcfdc1121869029da45192238e85cf44f0b232a6d9dc20e4fe6f4242a14e",
    },
];

pub(super) const QWEN_4BIT_FILES: &[PinnedSnapshotFile] = &[
    PinnedSnapshotFile {
        path: "config.json",
        bytes: 937,
        sha256: "507a6701220524eb8b283425bf0856a9ae4f21f4052e563896ddd668994b1dc7",
    },
    PinnedSnapshotFile {
        path: "model.safetensors",
        bytes: 968_080_210,
        sha256: "0e86d9677e519323849eac1bc272caae88567a481ff188c431f70be543d9995f",
    },
    PinnedSnapshotFile {
        path: "model.safetensors.index.json",
        bytes: 49_731,
        sha256: "1e3058d4ba4b04e4de35b74467725cbef90ff022198404218e48f21adc9cfa15",
    },
    PinnedSnapshotFile {
        path: "tokenizer_config.json",
        bytes: 9_706,
        sha256: "253153d0738ceb4c668d2eff957714dd2bea0b56de772a9fdccd96cbf517e6a0",
    },
];

pub(super) const QWEN_BF16_FILES: &[PinnedSnapshotFile] = &[
    PinnedSnapshotFile {
        path: "config.json",
        bytes: 784,
        sha256: "fa2aca3f3437d838672845487b8dd013b6a1f022daafb457bc9cb08564663ee4",
    },
    PinnedSnapshotFile {
        path: "model.safetensors",
        bytes: 3_441_185_437,
        sha256: "790a82e1fcbac0fa316e4c21cd025c0c39aa03e882e33e2b1dc39407c9629e4e",
    },
    PinnedSnapshotFile {
        path: "model.safetensors.index.json",
        bytes: 22_148,
        sha256: "d6a4b92a4ede4c18d5bbc8e5814e7a4b899a514262e74b92cedba90f0c3e836d",
    },
    PinnedSnapshotFile {
        path: "tokenizer_config.json",
        bytes: 9_706,
        sha256: "253153d0738ceb4c668d2eff957714dd2bea0b56de772a9fdccd96cbf517e6a0",
    },
];

pub const LLAMA_CANDIDATE: BenchmarkModelSpec = BenchmarkModelSpec {
    family: "llama",
    role: "candidate",
    repository: "mlx-community/Llama-3.2-3B-Instruct-4bit",
    revision: "7f0dc925e0d0afb0322d96f9255cfddf2ba5636e",
    architecture: "LlamaForCausalLM",
    model_type: "llama",
    native_context_tokens: 131_072,
    fit_window_tokens: None,
    quantized: true,
    required_files: LLAMA_4BIT_FILES,
};

pub const LLAMA_REFERENCE: BenchmarkModelSpec = BenchmarkModelSpec {
    family: "llama",
    role: "bf16-reference",
    repository: "mlx-community/Llama-3.2-3B-Instruct-bf16",
    revision: "6d88ba43024fef71b10e52e101c7cd4598322601",
    architecture: "LlamaForCausalLM",
    model_type: "llama",
    native_context_tokens: 131_072,
    fit_window_tokens: None,
    quantized: false,
    required_files: LLAMA_BF16_FILES,
};

pub const QWEN_CANDIDATE: BenchmarkModelSpec = BenchmarkModelSpec {
    family: "qwen",
    role: "candidate",
    repository: "mlx-community/Qwen3-1.7B-4bit",
    revision: "3b1b1768f8f8cf8351c712464f906e86c2b8269e",
    architecture: "Qwen3ForCausalLM",
    model_type: "qwen3",
    native_context_tokens: 40_960,
    fit_window_tokens: None,
    quantized: true,
    required_files: QWEN_4BIT_FILES,
};

pub const QWEN_REFERENCE: BenchmarkModelSpec = BenchmarkModelSpec {
    family: "qwen",
    role: "bf16-reference",
    repository: "mlx-community/Qwen3-1.7B-bf16",
    revision: "9cd6692855d3e06772228e9a962b2606359b2d24",
    architecture: "Qwen3ForCausalLM",
    model_type: "qwen3",
    native_context_tokens: 40_960,
    fit_window_tokens: None,
    quantized: false,
    required_files: QWEN_BF16_FILES,
};

// Epic 20669 (sc-20688): the 8B pair of each family, measured on the same row structure so
// compressed K8V8 can be qualified for them. Each candidate is the mlx-community 4-bit group-64
// conversion; each bf16 reference is that family's mlx-community bf16 conversion.
pub(super) const LLAMA8B_4BIT_FILES: &[PinnedSnapshotFile] = &[
    PinnedSnapshotFile {
        path: "config.json",
        bytes: 1097,
        sha256: "88804b1a541a86ce1b5b21dd0b38d95ee99006ef656a0b59ab267ecb0bce8b22",
    },
    PinnedSnapshotFile {
        path: "model.safetensors",
        bytes: 4_517_489_037,
        sha256: "192065799d1621df78b68274137974d3258c5dadce9ca71305ed014d997d67c4",
    },
    PinnedSnapshotFile {
        path: "model.safetensors.index.json",
        bytes: 52_381,
        sha256: "9a76e05055778bb04cacb7aff616b378da0e57c87a181c932037a943efba5997",
    },
    PinnedSnapshotFile {
        path: "tokenizer_config.json",
        bytes: 55_381,
        sha256: "f0f2d5fa9caf736f4184e3fe3316378290696f9ebd9f11ac431d48fbda92a11c",
    },
];

pub(super) const LLAMA8B_BF16_FILES: &[PinnedSnapshotFile] = &[
    PinnedSnapshotFile {
        path: "config.json",
        bytes: 944,
        sha256: "cc4b961ba4da639c74e00e59956dadbca2471bee85bccabecf9373bfcb5a47b3",
    },
    PinnedSnapshotFile {
        path: "model-00001-of-00004.safetensors",
        bytes: 5_295_466_428,
        sha256: "e8f8bd79686b2bf28b7f47826c14f038488c1e23311ffa64c5a6c84b9bd80a03",
    },
    PinnedSnapshotFile {
        path: "model-00002-of-00004.safetensors",
        bytes: 5_352_157_832,
        sha256: "0ba199f8cba7b46e09dad10ee239a2877d6467f9448c08db87f7a74e43ead599",
    },
    PinnedSnapshotFile {
        path: "model-00003-of-00004.safetensors",
        bytes: 4_362_258_775,
        sha256: "c37b5389bcd3d822c9a712b8fec3327585f760cc9b10081c81f9dfa9194d2794",
    },
    PinnedSnapshotFile {
        path: "model-00004-of-00004.safetensors",
        bytes: 1_050_673_279,
        sha256: "bfca38e630589bdba9d83040a771d63b36ea18ce5bce8f7c70493119a1f77fd5",
    },
    PinnedSnapshotFile {
        path: "model.safetensors.index.json",
        bytes: 25_125,
        sha256: "e60a22a6cec3b5ac425ffa46d368d64999c1f411d63f967f72861670a0745ba5",
    },
    PinnedSnapshotFile {
        path: "tokenizer_config.json",
        bytes: 55_351,
        sha256: "177c7b61e616fecb84c17ce0591acb92c6c4d60e9ac5ababfb940ff23bbcd424",
    },
];

pub(super) const QWEN8B_4BIT_FILES: &[PinnedSnapshotFile] = &[
    PinnedSnapshotFile {
        path: "config.json",
        bytes: 939,
        sha256: "e5485285fd7e289e76e9cffa112f6dc2e3426519082f7db9b69041589f81a218",
    },
    PinnedSnapshotFile {
        path: "model.safetensors",
        bytes: 4_607_835_174,
        sha256: "f2d29621aab300336ad645567ff38c42aac755513006ef4e8a579cf7ef5256d8",
    },
    PinnedSnapshotFile {
        path: "model.safetensors.index.json",
        bytes: 64_065,
        sha256: "3fb25463b4078b1fc27159daa605190029c2e965f533bf0b1b594f96cbfceb8a",
    },
    PinnedSnapshotFile {
        path: "tokenizer_config.json",
        bytes: 9_706,
        sha256: "253153d0738ceb4c668d2eff957714dd2bea0b56de772a9fdccd96cbf517e6a0",
    },
];

pub(super) const QWEN8B_BF16_FILES: &[PinnedSnapshotFile] = &[
    PinnedSnapshotFile {
        path: "config.json",
        bytes: 786,
        sha256: "5d758e724677ed4cefc9fe625717e29ee051df5511f89c803fe418c4d34f84f8",
    },
    PinnedSnapshotFile {
        path: "model-00001-of-00004.safetensors",
        bytes: 5_288_151_758,
        sha256: "511de1a92abea7eb36e57935cbdb9351ba0c1eb0941981127de92058d1cde22c",
    },
    PinnedSnapshotFile {
        path: "model-00002-of-00004.safetensors",
        bytes: 5_301_854_484,
        sha256: "59f362e0f24dac3fbc08c0aacc56bc50747fe819322387b134a9cba2d0fc3cfd",
    },
    PinnedSnapshotFile {
        path: "model-00003-of-00004.safetensors",
        bytes: 4_546_850_621,
        sha256: "d9eee2e91d2fcc18bf435d4464346a603d0a0727350767c9d5cec13b203f1df7",
    },
    PinnedSnapshotFile {
        path: "model-00004-of-00004.safetensors",
        bytes: 1_244_659_839,
        sha256: "3b37554fa92ed7cb0282698b5f490aaa537dafd528511bbb3e77cdf004862e00",
    },
    PinnedSnapshotFile {
        path: "model.safetensors.index.json",
        bytes: 34_485,
        sha256: "ea95c14009d02d9eedf3e96d83a51f3ea81802243e3de0ae3560b2b89230aae1",
    },
    PinnedSnapshotFile {
        path: "tokenizer_config.json",
        bytes: 9_706,
        sha256: "253153d0738ceb4c668d2eff957714dd2bea0b56de772a9fdccd96cbf517e6a0",
    },
];

/// Llama-3.1-8B's fit-boundary window. The compressed fit row's admission estimate is the
/// candidate estimate scaled by [`COMPRESSED_MEASURED_PEAK_SCALE_BPS`] (the measured A2 v5 peak
/// ratio). At the native 130,560-token row that is 59.2 GiB x 1.27 = 75.2 GiB, over the 68 GiB
/// child footprint cap of `.github/kv-poc/policies/llm.json`, so the row would be refused before
/// spawn. A 107,008-token window puts the fit row at 106,496 tokens: 50.0 GiB x 1.27 = 63.5 GiB,
/// 4.5 GiB under the cap; the next 2,048-token step (108,544) leaves 3.5 GiB. The other bands keep
/// the native window (memory-material 32,768). A dense (A1) row also loads the bf16 reference
/// (71.5 GiB here), which no fit row of at least 90% of this window keeps under the cap.
pub const LLAMA8B_FIT_WINDOW_TOKENS: u64 = 107_008;

pub const LLAMA8B_CANDIDATE: BenchmarkModelSpec = BenchmarkModelSpec {
    family: "llama8b",
    role: "candidate",
    repository: "mlx-community/Llama-3.1-8B-Instruct-4bit",
    revision: "90215b22ec18e72f623dde2ea7af4097025160e2",
    architecture: "LlamaForCausalLM",
    model_type: "llama",
    native_context_tokens: 131_072,
    fit_window_tokens: Some(LLAMA8B_FIT_WINDOW_TOKENS),
    quantized: true,
    required_files: LLAMA8B_4BIT_FILES,
};

pub const LLAMA8B_REFERENCE: BenchmarkModelSpec = BenchmarkModelSpec {
    family: "llama8b",
    role: "bf16-reference",
    repository: "mlx-community/Meta-Llama-3.1-8B-Instruct-bf16",
    revision: "f8311090f9ee47782b6f094984a20c856eb841d6",
    architecture: "LlamaForCausalLM",
    model_type: "llama",
    native_context_tokens: 131_072,
    fit_window_tokens: Some(LLAMA8B_FIT_WINDOW_TOKENS),
    quantized: false,
    required_files: LLAMA8B_BF16_FILES,
};

pub const QWEN8B_CANDIDATE: BenchmarkModelSpec = BenchmarkModelSpec {
    family: "qwen8b",
    role: "candidate",
    repository: "mlx-community/Qwen3-8B-4bit",
    revision: "545dc4251c05440727734bcd94334791f6ab0192",
    architecture: "Qwen3ForCausalLM",
    model_type: "qwen3",
    native_context_tokens: 40_960,
    fit_window_tokens: None,
    quantized: true,
    required_files: QWEN8B_4BIT_FILES,
};

pub const QWEN8B_REFERENCE: BenchmarkModelSpec = BenchmarkModelSpec {
    family: "qwen8b",
    role: "bf16-reference",
    repository: "mlx-community/Qwen3-8B-bf16",
    revision: "85dd0f16bfe491befbc9cf0b4e966664236e5050",
    architecture: "Qwen3ForCausalLM",
    model_type: "qwen3",
    native_context_tokens: 40_960,
    fit_window_tokens: None,
    quantized: false,
    required_files: QWEN8B_BF16_FILES,
};

/// The scheduled model families, in schedule order: each runs every context band on its own pinned
/// candidate/reference pair ([`benchmark_model`]).
pub const SC20671_FAMILIES: [&str; 4] = ["llama", "qwen", "llama8b", "qwen8b"];

/// The `'static` scheduled family named `family`, or `None` for an unscheduled name.
pub(super) fn scheduled_family(family: &str) -> Option<&'static str> {
    SC20671_FAMILIES
        .into_iter()
        .find(|candidate| *candidate == family)
}

/// The architecture family a session loaded from scheduled `family`'s pinned candidate reports
/// ([`crate::provider::LlamaProvider::campaign_family`]), derived from that spec's `model_type`:
/// `llama` and `llama8b` load as `"llama"`, `qwen` and `qwen8b` as `"qwen"`.
pub(super) fn coordinate_architecture_family(family: &str) -> Result<&'static str, String> {
    match benchmark_model(family, false)?.model_type {
        "llama" => Ok("llama"),
        "qwen3" => Ok("qwen"),
        other => Err(format!(
            "SC-20671 family {family:?} pins unsupported model_type {other:?}"
        )),
    }
}

pub fn benchmark_model(
    family: &str,
    reference: bool,
) -> Result<&'static BenchmarkModelSpec, String> {
    match (family, reference) {
        ("llama", false) => Ok(&LLAMA_CANDIDATE),
        ("llama", true) => Ok(&LLAMA_REFERENCE),
        ("qwen", false) => Ok(&QWEN_CANDIDATE),
        ("qwen", true) => Ok(&QWEN_REFERENCE),
        ("llama8b", false) => Ok(&LLAMA8B_CANDIDATE),
        ("llama8b", true) => Ok(&LLAMA8B_REFERENCE),
        ("qwen8b", false) => Ok(&QWEN8B_CANDIDATE),
        ("qwen8b", true) => Ok(&QWEN8B_REFERENCE),
        _ => Err(format!(
            "SC-20671 has no immutable benchmark model for family {family:?}"
        )),
    }
}

pub fn context_band_target(context_window: u64, context_band: &str) -> Result<u64, String> {
    if context_window < 1_024 {
        return Err("SC-20671 requires a context window of at least 1024 tokens".into());
    }
    let medium = (context_window / 16).clamp(128, 1_024);
    let memory_material = context_window / 4;
    let fit_by_ratio = context_window
        .checked_mul(FIT_BOUNDARY_MIN_CONTEXT_BPS)
        .and_then(|tokens| tokens.checked_add(9_999))
        .map(|tokens| tokens / 10_000)
        .ok_or("fit-boundary target overflows u64")?;
    let fit_boundary = context_window.saturating_sub(512).max(fit_by_ratio);
    if !(32 < medium && medium < memory_material && memory_material < fit_boundary) {
        return Err(format!(
            "loaded context window {context_window} cannot represent four distinct bands"
        ));
    }
    match context_band {
        "short" => Ok(32),
        "medium" => Ok(medium),
        "memory-material" => Ok(memory_material),
        "fit-boundary" => Ok(fit_boundary),
        _ => Err(format!("unknown context band {context_band}")),
    }
}

/// A spec's target for `context_band`: [`context_band_target`] of the native window, except the
/// fit-boundary band of a spec with a [`BenchmarkModelSpec::fit_window_tokens`] cap, which is that
/// window's fit target.
pub fn spec_band_target(spec: &BenchmarkModelSpec, context_band: &str) -> Result<u64, String> {
    match (context_band, spec.fit_window_tokens) {
        ("fit-boundary", Some(window)) if window <= spec.native_context_tokens => {
            context_band_target(window, context_band)
        }
        ("fit-boundary", Some(_)) => Err(format!(
            "{} fit window exceeds its native window",
            spec.repository
        )),
        _ => context_band_target(spec.native_context_tokens, context_band),
    }
}

/// The fit window of a `family` row measured at `context_window`: the family's
/// [`BenchmarkModelSpec::fit_window_tokens`] cap when that is its native window, else the window.
pub(super) fn row_fit_window(family: &str, context_window: u64) -> u64 {
    benchmark_model(family, false)
        .ok()
        .filter(|spec| spec.native_context_tokens == context_window)
        .and_then(|spec| spec.fit_window_tokens)
        .unwrap_or(context_window)
}

/// A recorded row's band target: [`context_band_target`] of its window, with the family's fit cap
/// ([`row_fit_window`]) on the fit-boundary band.
pub(super) fn row_band_target(
    family: &str,
    context_window: u64,
    context_band: &str,
) -> Result<u64, String> {
    if context_band == "fit-boundary" {
        context_band_target(row_fit_window(family, context_window), context_band)
    } else {
        context_band_target(context_window, context_band)
    }
}

/// [`spec_band_target`] of a scheduled family's candidate.
pub fn family_band_target(family: &str, context_band: &str) -> Result<u64, String> {
    spec_band_target(benchmark_model(family, false)?, context_band)
}

/// Live tokens of a compressed row's forced continuation: the kernel prompt plus its continuation,
/// which a fit-boundary row shortens to the native window.
pub(super) fn forced_continuation_live_tokens(
    kernel_prompt_tokens: u64,
    native_context_tokens: u64,
) -> u64 {
    kernel_prompt_tokens
        + FORCED_CONTINUATION_TOKENS.min(native_context_tokens.saturating_sub(kernel_prompt_tokens))
}

/// Compute a conservative total-live-token bound using only the pinned tokenizer and source
/// fixture construction. Product chat rendering is separately capped by the native window; the
/// prefix/batch paths bypass that renderer, so their exact raw prompt lengths are checked here.
/// A lifecycle can hold one prefix cache alongside one full native-window request, while a batch
/// can hold two prefix-length lanes. This is checked before any weights or Metal model is loaded.
pub(super) fn preflight_total_live_tokens(
    snapshot: &Path,
    spec: &BenchmarkModelSpec,
    coordinate: &Coordinate,
    prompt: &str,
    compressed: bool,
) -> Result<(u64, u64), String> {
    let tokenizer = Tokenizer::from_file(snapshot.join("tokenizer.json"))
        .map_err(|e| format!("load pinned tokenizer for safety preflight: {e}"))?;
    let target = spec_band_target(spec, coordinate.context_band)?;
    let header = format!("SC20671-CONTEXT-BAND-{}", coordinate.context_band);
    let token_count = |text: &str| -> Result<u64, String> {
        u64::try_from(
            tokenizer
                .encode(text, false)
                .map_err(|e| e.to_string())?
                .len(),
        )
        .map_err(|_| "preflight token count overflows u64".into())
    };
    let mut lower = 0_usize;
    let mut upper = usize::try_from(target).map_err(|_| "target overflows usize")?;
    while lower < upper {
        let midpoint = lower + (upper - lower).div_ceil(2);
        let candidate = format!("{header}{}", " context".repeat(midpoint));
        if token_count(&candidate)? <= target {
            lower = midpoint;
        } else {
            upper = midpoint - 1;
        }
    }
    let payload = format!("{header}{}", " context".repeat(lower));
    // The multi-turn fixture's payload: the same search, capped (fit-boundary) by
    // `multi_turn_payload_target`.
    let multi_turn_target = multi_turn_payload_target(spec.native_context_tokens, target)?;
    let (mut lower, mut upper) = (0_usize, lower);
    while lower < upper {
        let midpoint = lower + (upper - lower).div_ceil(2);
        let candidate = format!("{header}{}", " context".repeat(midpoint));
        if token_count(&candidate)? <= multi_turn_target {
            lower = midpoint;
        } else {
            upper = midpoint - 1;
        }
    }
    let multi_turn_prompt = format!(
        "{prompt}\n{header}{}\nRepeat the stable baseline fact.",
        " context".repeat(lower)
    );
    let needle = "SC20671-NUMERIC-NEEDLE-9b7a2e";
    let prompts = [
        format!("{prompt}\n{payload}\nReturn a concise deterministic answer."),
        format!("{prompt}\n{payload}\nCall record_baseline_fact with fact exactly `SC20671 structured fixture`."),
        needle_fixture_prompt(prompt, &payload, needle),
        format!("{prompt}\n{payload}\nRepeat the stable baseline fact."),
    ];
    let prompt_tokens = prompts
        .iter()
        .map(|text| token_count(text))
        .collect::<Result<Vec<_>, _>>()?;
    let max_prefix = prompt_tokens
        .iter()
        .copied()
        .max()
        .ok_or("no fixture prompts")?;
    // Each candidate repeat's steady decode holds the kernel fixture's raw context plus the fixed
    // decode length in its own cache, alone (every fixture cache has been released).
    let steady_decode = prompt_tokens[0]
        .checked_add(STEADY_DECODE_TOKENS)
        .ok_or("steady-decode live context overflows u64")?;
    let direct_request = max_prefix
        .checked_add(2)
        .ok_or("direct request token count overflows")?;
    if max_prefix > spec.native_context_tokens.saturating_sub(256) {
        return Err(format!(
            "{} fixture prompt exceeds the producer's native-context reserve before model load",
            coordinate_slug(coordinate)
        ));
    }
    // Direct prefix and batch paths bypass the chat renderer but generate at most one/two tokens.
    if direct_request > spec.native_context_tokens {
        return Err(format!(
            "{} prefix/batch prompt exceeds native context before model load",
            coordinate_slug(coordinate)
        ));
    }
    let rendered = prompts
        .iter()
        .enumerate()
        .map(|(index, text)| {
            let tools = if index == 1 {
                vec![structured_fixture_tool()]
            } else {
                Vec::new()
            };
            crate::provider::campaign_preflight_request_tokens(
                snapshot,
                &fixture_request(text.clone(), tools),
            )
            .map_err(|e| e.to_string())
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .max()
        .ok_or("no rendered requests")?;
    // The multi-turn fixture (contract v4) holds turn 1's stored prompt and answer beside turn
    // 2's request cache: turn 2 renders turn 1's conversation, an answer of at most
    // `FIXTURE_MAX_NEW_TOKENS` tokens (bounded here by a placeholder answer of that many words plus
    // a re-tokenization margin), and the follow-up, then decodes its natural answer or, on a
    // compressed row, its forced continuation.
    let turn1_rendered = crate::provider::campaign_preflight_request_tokens(
        snapshot,
        &fixture_request(multi_turn_prompt.clone(), Vec::new()),
    )
    .map_err(|e| e.to_string())?;
    let mut turn2_request = fixture_request(multi_turn_prompt, Vec::new());
    turn2_request.messages.push(Message::text(
        Role::Assistant,
        " context".repeat(FIXTURE_MAX_NEW_TOKENS as usize),
    ));
    turn2_request
        .messages
        .push(Message::text(Role::User, MULTI_TURN_FIXTURE_FOLLOW_UP));
    let turn2_rendered =
        crate::provider::campaign_preflight_request_tokens(snapshot, &turn2_request)
            .map_err(|e| e.to_string())?
            .checked_add(MULTI_TURN_ANSWER_RETOKENIZATION_MARGIN)
            .ok_or("multi-turn live context overflows u64")?;
    // Contract v4: turn 2 must leave the full turn-2 forced continuation, on every row (the
    // fixture is the same for dense and compressed rows), or the row is refused before load.
    if turn2_rendered + FORCED_CONTINUATION_TOKENS > spec.native_context_tokens {
        return Err(format!(
            "{} multi-turn fixture turn 2 ({turn2_rendered} tokens) leaves fewer than the \
             {FORCED_CONTINUATION_TOKENS}-token forced continuation in the {}-token window",
            coordinate_slug(coordinate),
            spec.native_context_tokens
        ));
    }
    let turn2_decode = if compressed {
        FORCED_CONTINUATION_TOKENS
    } else {
        u64::from(FIXTURE_MAX_NEW_TOKENS)
    };
    let multi_turn = [
        turn1_rendered,
        u64::from(FIXTURE_MAX_NEW_TOKENS),
        turn2_rendered,
        turn2_decode,
    ]
    .into_iter()
    .try_fold(0_u64, u64::checked_add)
    .ok_or("multi-turn live context overflows u64")?;
    let cancellation = crate::provider::campaign_preflight_cancellation_tokens(snapshot)
        .map_err(|e| e.to_string())?;
    if rendered > spec.native_context_tokens || cancellation > spec.native_context_tokens {
        return Err(format!(
            "{} rendered request exceeds native context before model load",
            coordinate_slug(coordinate)
        ));
    }
    let max_request = rendered.max(direct_request).max(cancellation);
    let prefix_plus_request = max_prefix
        .checked_add(1)
        .and_then(|prefix| prefix.checked_add(rendered.max(cancellation)))
        .ok_or("total live context overflows u64")?;
    let batch = max_prefix
        .checked_add(2)
        .and_then(|tokens| tokens.checked_mul(2))
        .ok_or("batch live context overflows u64")?;
    let total = if coordinate.request_mode == "supported-batch" {
        if coordinate.prefill_mode == "chunked" {
            prefix_plus_request.max(batch)
        } else {
            max_request.max(batch)
        }
    } else if coordinate.prefill_mode == "chunked" {
        prefix_plus_request
    } else {
        max_request
    };
    // A compressed row's forced continuation holds the kernel context plus its continuation (at
    // most the native window) in its own request-scoped cache.
    let forced_continuation = if compressed {
        forced_continuation_live_tokens(prompt_tokens[0], spec.native_context_tokens)
    } else {
        0
    };
    Ok((
        total
            .max(steady_decode)
            .max(forced_continuation)
            .max(multi_turn),
        max_request.max(turn2_rendered),
    ))
}

/// The dtype every campaign role's dense K/V is cached in: the causal loader's compute dtype. The
/// loader holds stored quantized scales/biases in the compute dtype (sc-20671), so a stored F16/F32
/// scale no longer promotes the activations — and the cache — to F32.
pub const DENSE_KV_COMPUTE_DTYPE: mlx_rs::Dtype = crate::models::CausalLm::COMPUTE_DTYPE;

/// Scalar width of [`DENSE_KV_COMPUTE_DTYPE`]. [`ProductObserver`] and
/// [`validate_receipt_semantics`] refuse any other observed width rather than record a silently
/// widened dense baseline.
pub const DENSE_KV_COMPUTE_ELEMENT_BYTES: u64 =
    crate::primitives::dtype_bytes(DENSE_KV_COMPUTE_DTYPE);

/// Validate the pinned decoder projection inventory without loading MLX arrays and return the
/// dense K/V scalar width it produces. Every supported stored scale dtype (BF16/F16/F32) is cast to
/// the BF16 compute dtype at load, and dense projections are cast to BF16 too, so the width is the
/// compute dtype's whatever the snapshot stores — never the widest stored scale (the pre-sc-20671
/// F32-promoted path). Both parent preflight and worker validate the exact pinned snapshot
/// inventory before this call.
pub(super) fn pinned_dense_kv_element_bytes(
    spec: &BenchmarkModelSpec,
    snapshot: &Path,
    layers: u64,
) -> Result<u64, String> {
    let mut dtypes = std::collections::HashMap::<String, String>::new();
    for required in spec
        .required_files
        .iter()
        .filter(|file| file.path.ends_with(".safetensors"))
    {
        let path = snapshot.join(required.path);
        let mut file = File::open(&path)
            .map_err(|e| format!("read pinned weight header {}: {e}", path.display()))?;
        let mut prefix = [0_u8; 8];
        file.read_exact(&mut prefix)
            .map_err(|e| format!("read pinned weight header length {}: {e}", path.display()))?;
        let len = u64::from_le_bytes(prefix);
        if len > 64 * 1024 * 1024 || len > required.bytes.saturating_sub(8) {
            return Err("pinned safetensors header length is invalid".into());
        }
        let mut bytes = vec![0_u8; len as usize];
        file.read_exact(&mut bytes)
            .map_err(|e| format!("read pinned weight header {}: {e}", path.display()))?;
        let header: serde_json::Value = serde_json::from_slice(&bytes)
            .map_err(|e| format!("parse pinned weight header: {e}"))?;
        for (key, tensor) in header
            .as_object()
            .ok_or("pinned safetensors header is not an object")?
        {
            if key == "__metadata__" {
                continue;
            }
            let dtype = tensor
                .get("dtype")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| format!("pinned tensor {key} lacks dtype"))?;
            if dtypes.insert(key.clone(), dtype.to_owned()).is_some() {
                return Err(format!("duplicate pinned tensor {key}"));
            }
        }
    }
    for layer in 0..layers {
        for projection in [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ] {
            let stem = format!("model.layers.{layer}.{projection}");
            let weight = dtypes
                .get(&format!("{stem}.weight"))
                .ok_or_else(|| format!("pinned decoder lacks {stem}.weight"))?;
            match dtypes.get(&format!("{stem}.scales")) {
                Some(scales) => {
                    let biases = dtypes
                        .get(&format!("{stem}.biases"))
                        .ok_or_else(|| format!("pinned decoder lacks {stem}.biases"))?;
                    if !spec.quantized || weight != "U32" || scales != biases {
                        return Err(format!("unsupported pinned quantized projection {stem}"));
                    }
                    // Held in the BF16 compute dtype at load whatever the stored width.
                    if !matches!(scales.as_str(), "BF16" | "F16" | "F32") {
                        return Err(format!("unsupported pinned scale dtype for {stem}"));
                    }
                }
                None => {
                    if !matches!(weight.as_str(), "BF16" | "F16" | "F32")
                        || dtypes.contains_key(&format!("{stem}.biases"))
                    {
                        return Err(format!("unsupported pinned dense projection {stem}"));
                    }
                }
            }
        }
    }
    Ok(DENSE_KV_COMPUTE_ELEMENT_BYTES)
}

/// The fail-closed reason for an observed dense K/V width that is not the compute dtype's.
pub(super) fn dense_kv_width_refusal(observed: u64) -> String {
    let observed_dtype = match observed {
        1 => "an 8-bit",
        2 => "a 16-bit float",
        4 => "Float32",
        8 => "Float64",
        _ => "an unknown",
    };
    format!(
        "dense KV observed as {observed_dtype} dtype ({observed} bytes/element), expected the \
         {DENSE_KV_COMPUTE_DTYPE:?} compute dtype ({DENSE_KV_COMPUTE_ELEMENT_BYTES} \
         bytes/element); the loader promoted the cache (sc-20671), so this row is refused rather \
         than recorded as a widened dense baseline"
    )
}

/// A deliberately conservative planning floor, not a proven process peak. The two-times
/// checkpoint load reservation matches this model type's load admission; the pinned projection
/// metadata supplies role-specific dense KV width. The live KV is the total-live bound's (a stored
/// prefix alongside one full request, which is all the prefix path holds since sc-20671). The
/// request's prefill activations — one decoder layer's projections, MLP tensors and residuals
/// across the whole request, priced exactly as the product's tiled request admission prices them
/// ([`core_llm::tiled_prefill_activation_bytes`]) — scale with the request like the KV does and
/// were missing before sc-20671 (~12 GiB for the 3B bf16 reference at the fit boundary). The
/// actual lazy graph may retain more, especially at long context, so the floor is only a
/// pre-spawn refusal (a row whose floor already exceeds the child cap cannot fit) and the
/// estimate recorded in the receipt. Admission itself is runtime-guarded: see
/// [`runtime_guarded_admission`].
pub(crate) fn static_role_footprint_budget(
    spec: &BenchmarkModelSpec,
    snapshot: &Path,
    total_live_tokens: u64,
    request_tokens: u64,
) -> Result<u64, String> {
    let payload = spec
        .required_files
        .iter()
        .filter(|file| file.path.ends_with(".safetensors"))
        .try_fold(0_u64, |sum, file| sum.checked_add(file.bytes))
        .ok_or("pinned checkpoint payload overflows")?;
    if payload == 0 {
        return Err("pinned model has no checkpoint payload".into());
    }
    let config: serde_json::Value =
        serde_json::from_slice(&fs::read(snapshot.join("config.json")).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let positive = |key: &str| {
        config
            .get(key)
            .and_then(serde_json::Value::as_u64)
            .filter(|value| *value > 0)
            .ok_or_else(|| format!("pinned model lacks positive {key} for footprint preflight"))
    };
    if config
        .get("torch_dtype")
        .and_then(serde_json::Value::as_str)
        != Some("bfloat16")
    {
        return Err("pinned model loader input dtype is not BF16".into());
    }
    let layers = positive("num_hidden_layers")?;
    let kv_heads = positive("num_key_value_heads")?;
    let query_heads = positive("num_attention_heads")?;
    let hidden_size = positive("hidden_size")?;
    // The product loader's default when the config omits `head_dim` (Llama 3.1 does).
    let head_dim = match config.get("head_dim") {
        None => (hidden_size % query_heads == 0)
            .then(|| hidden_size / query_heads)
            .ok_or("pinned model hidden_size is not a multiple of num_attention_heads")?,
        Some(_) => positive("head_dim")?,
    };
    let intermediate_size = positive("intermediate_size")?;
    let vocab_size = positive("vocab_size")?;
    let element_bytes = pinned_dense_kv_element_bytes(spec, snapshot, layers)?;
    let kv = dense_kv_bytes(
        1,
        layers,
        kv_heads,
        total_live_tokens,
        head_dim,
        element_bytes,
    )?;
    let tile = [3_u64, 8, request_tokens, query_heads, 4]
        .into_iter()
        .try_fold(1_u64, |value, factor| value.checked_mul(factor))
        .ok_or("fused prefill tile footprint overflows")?;
    let activations = core_llm::tiled_prefill_activation_bytes(
        request_tokens,
        core_llm::LlmMemoryGeometry {
            query_heads,
            kv_heads,
            head_dim,
            layers,
            element_bytes,
            score_element_bytes: 4,
            hidden_size,
            intermediate_size,
            vocab_size,
            recurrent_bytes: 0,
        },
    )
    .ok_or("prefill activation footprint overflows")?;
    payload
        .checked_mul(2)
        .and_then(|bytes| bytes.checked_add(kv))
        .and_then(|bytes| bytes.checked_add(tile))
        .and_then(|bytes| bytes.checked_add(activations))
        .ok_or("static model-load plus KV/working budget overflows".into())
}

/// Measured whole-process peak over the static candidate estimate of a compressed row, in basis
/// points: the largest ratio of the A2 v5 receipts (run 37004025116, group-affine-8, inference
/// 0030a60b8; receipt `memory.phaseSamples[*].physFootprintPeakBytes` over the candidate-only
/// estimate this source computes for the same row): qwen-fit-boundary 17,960,978,184 B over
/// 14,149,222,820 B is 1.2694; llama-fit-boundary 56,136,371,440 B over 46,768,365,740 B is
/// 1.2003; llama/qwen memory-material 1.0715 / 1.0400; the short and medium rows sit below 1 (the
/// group-affine-4 run's peaks are lower still). Rounded up to 1.27.
pub const COMPRESSED_MEASURED_PEAK_SCALE_BPS: u64 = 12_700;

/// The row's static floor over the roles its worker loads: a dense row loads the candidate then
/// the bf16 reference (the larger prices the row); a compressed row's quality reference is the
/// dense-KV arm on the SAME candidate weights, so only the candidate is ever loaded and the bf16
/// reference is never priced. The candidate-only source estimate sat below the measured peak of
/// the long compressed rows, so it is scaled by [`COMPRESSED_MEASURED_PEAK_SCALE_BPS`].
pub(super) fn static_row_footprint_budget(
    coordinate: &Coordinate,
    candidate_snapshot: &Path,
    reference_snapshot: &Path,
    total_live_tokens: u64,
    request_tokens: u64,
    policy: &CampaignSafetyPolicy,
    compressed: bool,
) -> Result<u64, String> {
    let candidate = static_role_footprint_budget(
        benchmark_model(coordinate.family, false)?,
        candidate_snapshot,
        total_live_tokens,
        request_tokens,
    )?;
    let required = if compressed {
        candidate
            .checked_mul(COMPRESSED_MEASURED_PEAK_SCALE_BPS)
            .map(|bytes| bytes.div_ceil(10_000))
            .ok_or("scaled compressed footprint budget overflows")?
    } else {
        candidate.max(static_role_footprint_budget(
            benchmark_model(coordinate.family, true)?,
            reference_snapshot,
            total_live_tokens,
            request_tokens,
        )?)
    };
    if required > policy.child_footprint_cap_bytes {
        return Err(format!(
            "{} needs at least {required} bytes of source-based model-load plus KV/working budget; child footprint cap {} refuses before spawn",
            coordinate_slug(coordinate), policy.child_footprint_cap_bytes,
        ));
    }
    Ok(required)
}

pub(super) fn static_row_requirements(
    coordinate: &Coordinate,
    candidate_snapshot: &Path,
    reference_snapshot: &Path,
    prompt: &str,
    policy: &CampaignSafetyPolicy,
    compressed: bool,
) -> Result<(u64, u64, u64), String> {
    let candidate = benchmark_model(coordinate.family, false)?;
    let reference = benchmark_model(coordinate.family, true)?;
    let candidate_tokens = preflight_total_live_tokens(
        candidate_snapshot,
        candidate,
        coordinate,
        prompt,
        compressed,
    )?;
    let reference_tokens = preflight_total_live_tokens(
        reference_snapshot,
        reference,
        coordinate,
        prompt,
        compressed,
    )?;
    let total = candidate_tokens.0.max(reference_tokens.0);
    let max_request = candidate_tokens.1.max(reference_tokens.1);
    if max_request > policy.max_request_tokens {
        return Err(format!(
            "{} requires actual request ceiling {max_request}; policy refuses before model load",
            coordinate_slug(coordinate)
        ));
    }
    if total > policy.max_context_tokens {
        return Err(format!(
            "{} requires conservative total-live ceiling {total}; policy refuses before model load",
            coordinate_slug(coordinate)
        ));
    }
    let known_footprint_budget = static_row_footprint_budget(
        coordinate,
        candidate_snapshot,
        reference_snapshot,
        total,
        max_request,
        policy,
        compressed,
    )?;
    Ok((total, max_request, known_footprint_budget))
}

/// The live tokens one noise-floor worker holds at its peak: the live-token bound of a compressed
/// row's same-weights dense reference arm, which runs exactly nf's passes (the kernel forced
/// continuation to 1024 tokens or the window, and the multi-turn fixture whose stored turn 1 sits
/// beside turn 2's request cache through a 1024-token forced turn-2 continuation). That bound
/// already counts the stored prefix beside the request, so nothing is stacked on it.
pub(super) fn noise_floor_live_tokens(
    snapshot: &Path,
    spec: &BenchmarkModelSpec,
    coordinate: &Coordinate,
    prompt: &str,
) -> Result<(u64, u64), String> {
    preflight_total_live_tokens(snapshot, spec, coordinate, prompt, true)
}

/// The static footprint of one noise-floor worker: one dense candidate session over
/// [`noise_floor_live_tokens`]: model load, that live K/V, the prefill tile and activations.
pub(super) fn noise_floor_footprint_budget(
    spec: &BenchmarkModelSpec,
    snapshot: &Path,
    total_live_tokens: u64,
    request_tokens: u64,
) -> Result<u64, String> {
    static_role_footprint_budget(spec, snapshot, total_live_tokens, request_tokens)
}

/// The requirements of one noise-floor row: what its worker actually holds, one dense session of
/// the candidate at a time (never the bf16 reference, which it does not load) over
/// [`noise_floor_live_tokens`]. The chunked-prefill control's cache is sized once
/// ([`crate::provider`]), so that one-session budget prices every pass of the row.
pub(super) fn noise_floor_row_requirements(
    coordinate: &Coordinate,
    snapshot: &Path,
    prompt: &str,
    policy: &CampaignSafetyPolicy,
) -> Result<(u64, u64, u64), String> {
    let spec = benchmark_model(coordinate.family, false)?;
    let (total, max_request) = noise_floor_live_tokens(snapshot, spec, coordinate, prompt)?;
    if max_request > policy.max_request_tokens || total > policy.max_context_tokens {
        return Err(format!(
            "{} requires request ceiling {max_request} and total-live ceiling {total}; policy refuses before model load",
            coordinate_slug(coordinate)
        ));
    }
    let budget = noise_floor_footprint_budget(spec, snapshot, total, max_request)?;
    if budget > policy.child_footprint_cap_bytes {
        return Err(format!(
            "{} needs at least {budget} bytes of one dense session; child footprint cap {} refuses before spawn",
            coordinate_slug(coordinate),
            policy.child_footprint_cap_bytes,
        ));
    }
    Ok((total, max_request, budget))
}
