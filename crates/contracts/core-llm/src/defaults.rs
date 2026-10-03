//! **The per-backend decode defaults table** (epic sc-24432 E5) — the one place every decode
//! optimization's default is set, one row per backend: [`MLX`], [`CANDLE_CUDA`], [`CANDLE_METAL`]
//! and [`CANDLE_CPU`] (Candle on Metal has its own row because it shares neither the CUDA kernels
//! nor the CPU's costs).
//!
//! Every default is **on** unless a measured regression or a correctness limit justifies off, and
//! that justification sits next to the value (`// justification:`). A value the epic's terminal
//! benchmark campaign (sc-24446) measured is marked `MEASURED` and names its evidence: the on-vs-off
//! pairs of the compare reports `mlx-final-compare.md` (campaigns `mlx-campaign-4` and
//! `mlx-campaign-6`) and `cuda-b4-compare.md` (campaign `cuda-campaign-b4`, CI run 37125806675),
//! written by `scripts/release/speculative_bench_campaign.py compare`. A row the campaign had no
//! host for (Candle on Metal, Candle on the CPU) follows the CUDA row's measured decision and says
//! so. A field the backend has no such path for is `false` with the reason `n/a`; a field marked
//! `fixed` names a path the backend always takes (nothing reads the value, because there is no
//! alternative to switch to).
//!
//! The backends read their defaults from this table — process switches take their unset state
//! from it ([`ProcessSwitch`](crate::switch::ProcessSwitch)), a provider applies
//! [`DecodeDefaults::speculative`] to a request that leaves the speculative option unset
//! ([`TextLlmRequest::speculative_or`](crate::TextLlmRequest::speculative_or)) and a load sizes
//! its prefix cache from [`DecodeDefaults::prefix_cache_bytes`]. An explicit setting — a request
//! field, a `LoadSpec` field, a process environment variable or a runtime override — always wins
//! over the table.

use crate::request::{Speculative, SpeculativeProposer};
use crate::speculative::{
    DRAFT_MODEL_RECOMMENDED_DEPTH, MTP_RECOMMENDED_DEPTH, PROMPT_LOOKUP_RECOMMENDED_DEPTH,
};

/// A decode backend the defaults table has a row for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DecodeBackend {
    /// `mlx-llm` on Apple silicon.
    Mlx,
    /// `candle-llm` on a CUDA device.
    CandleCuda,
    /// `candle-llm` on a Metal device.
    CandleMetal,
    /// `candle-llm` on the CPU.
    CandleCpu,
}

impl DecodeBackend {
    /// Every backend, in table order.
    pub const ALL: [DecodeBackend; 4] = [
        DecodeBackend::Mlx,
        DecodeBackend::CandleCuda,
        DecodeBackend::CandleMetal,
        DecodeBackend::CandleCpu,
    ];

    /// The stable lower-case label (`mlx`, `candle_cuda`, `candle_metal`, `candle_cpu`).
    pub const fn label(self) -> &'static str {
        match self {
            DecodeBackend::Mlx => "mlx",
            DecodeBackend::CandleCuda => "candle_cuda",
            DecodeBackend::CandleMetal => "candle_metal",
            DecodeBackend::CandleCpu => "candle_cpu",
        }
    }

    /// This backend's row of the table.
    pub const fn defaults(self) -> &'static DecodeDefaults {
        match self {
            DecodeBackend::Mlx => &MLX,
            DecodeBackend::CandleCuda => &CANDLE_CUDA,
            DecodeBackend::CandleMetal => &CANDLE_METAL,
            DecodeBackend::CandleCpu => &CANDLE_CPU,
        }
    }
}

/// The speculative depth each proposer runs at under [`Speculative::Auto`] (and advertises as its
/// `recommended_depth`), clamped to the model's backend-true maximum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RecommendedDepths {
    /// The checkpoint's MTP head.
    pub mtp: u32,
    /// Prompt lookup.
    pub prompt_lookup: u32,
    /// A draft model.
    pub draft_model: u32,
}

impl RecommendedDepths {
    /// The recommended depth for `proposer`.
    pub const fn get(&self, proposer: SpeculativeProposer) -> u32 {
        match proposer {
            SpeculativeProposer::Mtp => self.mtp,
            SpeculativeProposer::PromptLookup => self.prompt_lookup,
            SpeculativeProposer::DraftModel => self.draft_model,
        }
    }
}

/// The proposer-intrinsic recommended depths every backend's row uses (the upstream checkpoints'
/// own recommendation for MTP; the n-gram and draft-model depths sc-24433 / sc-24436 chose).
const RECOMMENDED_DEPTHS: RecommendedDepths = RecommendedDepths {
    mtp: MTP_RECOMMENDED_DEPTH,
    prompt_lookup: PROMPT_LOOKUP_RECOMMENDED_DEPTH,
    draft_model: DRAFT_MODEL_RECOMMENDED_DEPTH,
};

/// The prefix-cache budget every row reserves when `LoadSpec::prefix_cache_bytes` is unset: 1 GiB,
/// clamped to what the load's own admission leaves
/// ([`prefix_cache_budget`](crate::prefix::prefix_cache_budget)).
const PREFIX_CACHE_BYTES: u64 = 1 << 30;

/// One backend's row: the default of every decode optimization on that backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DecodeDefaults {
    /// The backend this row is for.
    pub backend: DecodeBackend,
    /// The speculative option a request that leaves it unset runs with (legacy `mtp` unset too).
    pub speculative: Speculative,
    /// Per-proposer depth for [`Speculative::Auto`] and the advertised `recommended_depth`.
    pub recommended_depths: RecommendedDepths,
    /// The cross-turn prefix-cache budget a load reserves when `LoadSpec::prefix_cache_bytes` is
    /// unset (`0` disables the cache).
    pub prefix_cache_bytes: u64,
    /// Whether the decode step is captured and replayed as a CUDA graph.
    pub cuda_graphs: bool,
    /// Whether decoders stage positions on the device (and take the length-aware
    /// `decode_attention` kernel on CUDA).
    pub device_positions: bool,
    /// Whether the token-at-a-time loop enqueues step `t + 1` before reading step `t` back.
    pub pipelining: bool,
    /// Whether tokens are drawn on the device (argmax / the device sampler) rather than by the
    /// host reference.
    pub device_sampler: bool,
    /// Whether a decode-sized MoE step dispatches its experts from device-resident routes (the
    /// indexed / gathered dispatch) rather than reading the routes back and grouping on the host.
    pub moe_device_dispatch: bool,
    /// Whether the fused decode primitives (RMSNorm, RMSNorm+residual, SwiGLU, RoPE) run.
    pub fused_kernels: bool,
    /// Whether the Prism block-Hadamard rotation runs as one fused kernel.
    pub fused_rotation: bool,
    /// Whether the gated-delta (DeltaNet) recurrence runs as a fused GPU kernel.
    pub gdn_kernel: bool,
    /// Whether a long gated-delta prefill runs in the chunkwise-parallel form.
    pub gdn_chunked_prefill: bool,
    /// Whether decode-sized NVFP4 projections take the fused NVFP4 GEMV rather than cuBLASLt.
    pub nvfp4_gemv: bool,
    /// Whether attention routes each shape onto the fused MLX SDPA kernel that serves it
    /// (sc-24442) rather than the pre-sc-24442 fixed 8-row tiling.
    pub sdpa_kernel_routing: bool,
}

/// **MLX** (`mlx-llm`, Apple silicon).
pub const MLX: DecodeDefaults = DecodeDefaults {
    backend: DecodeBackend::Mlx,
    // justification: off, MEASURED (mlx-final-compare.md, mlx-campaign-6): every proposer is
    // greedy-exact (E1), but `auto` vs `off` regresses decode beyond the noise band on Qwen3-8B
    // (chat -19.0%) and the Gemma 4 enhancer (chat -3.9%), so a request opts in per family.
    speculative: Speculative::Off,
    // justification: proposer-intrinsic depths (MTP: the checkpoints' upstream recommendation;
    // prompt lookup / draft model: sc-24433 / sc-24436), clamped per model to the backend max.
    recommended_depths: RECOMMENDED_DEPTHS,
    // justification: on, MEASURED (mlx-final-compare.md, mlx-campaign-6): no on-vs-0 regression
    // on any family (200 judged, 0 regression); a hit cuts TTFT 19.5–87.7%. An exact cross-turn
    // reuse, admission-clamped (E7), so it can never refuse or overrun a load.
    prefix_cache_bytes: PREFIX_CACHE_BYTES,
    // justification: n/a — MLX has no CUDA graphs.
    cuda_graphs: false,
    // justification: fixed — MLX decoders always read positions from device RoPE tables
    // (sc-24442); there is no host-position path to switch to.
    device_positions: true,
    // justification: on (sc-24439) — token-identical to the unpipelined loop by construction (the
    // look-ahead step is enqueued on the unread device token; E1 parity suite) and it hides the
    // per-token read-back. MEASURED (mlx-final-compare.md, mlx-campaign-6, Qwen3-8B): no decode
    // regression on vs off (40 judged: 20 pass, 19 inconclusive); the one flagged cell, a TTFT
    // (chat +37.7%, band ±35.2%), was token 0 waiting behind step 1's build, which the loop now
    // delivers first (sc-24446). Runtime switch `MLX_LLM_PIPELINING`.
    pipelining: true,
    // justification: on (sc-24439) — greedy is the device argmax (bit-identical); stochastic draws
    // keep the target distribution (same weights rule as the host reference). Runtime switch
    // `MLX_LLM_DEVICE_SAMPLER`.
    device_sampler: true,
    // justification: fixed — MLX always dispatches routed experts with `gather_mm` / `gather_qmm`
    // on device routes (sc-24440); there is no host-grouped path.
    moe_device_dispatch: true,
    // justification: fixed — MLX's fused `fast::` kernels are the only norm / RoPE path.
    fused_kernels: true,
    // justification: on (sc-24444) — the fused Metal rotation is bit-identical to the op chain
    // by construction. Runtime switch `MLX_LLM_FUSED_ROTATION`.
    fused_rotation: true,
    // justification: on (sc-24443) — the fused Metal recurrence matches the op-by-op reference
    // (parity suite) and replaces a per-token op chain. Runtime switch `MLX_LLM_GDN_KERNEL`.
    gdn_kernel: true,
    // justification: on (sc-24443) — the chunkwise form serves long prefills off the GPU kernel
    // (a CPU stream, or `MLX_LLM_GDN_KERNEL` off), matching the reference within its tolerance.
    gdn_chunked_prefill: true,
    // justification: n/a — NVFP4 is a CUDA format.
    nvfp4_gemv: false,
    // justification: on (sc-24442) — every shape goes to the fused kernel that serves it; the
    // speculative max depth (`vector_verify_rows`) is derived from this routing, so turning it off
    // also invalidates the advertised MLX depths.
    sdpa_kernel_routing: true,
};

/// **Candle on CUDA** (`candle-llm`, `cuda` feature, CUDA device).
pub const CANDLE_CUDA: DecodeDefaults = DecodeDefaults {
    backend: DecodeBackend::CandleCuda,
    // justification: off, MEASURED (cuda-b4-compare.md, cuda-campaign-b4): greedy-exact, but
    // `auto` vs `off` regresses decode beyond the noise band on Bonsai (graphs on: -5.6..-18.3%),
    // so a request opts in.
    speculative: Speculative::Off,
    // justification: proposer-intrinsic depths, clamped per model to the backend max.
    recommended_depths: RECOMMENDED_DEPTHS,
    // justification: on, MEASURED (cuda-b4-compare.md, cuda-campaign-b4): a hit cuts TTFT
    // 15.1–96.9%; a miss leaves decode within -2.1..+13.1% and costs TTFT 0–4% on every cell with
    // a tight band (one Qwen3-8B cell reads +21.6% inside a ±54.6% band). The tool's off verdict
    // rests on two cells barely past their band (a +2.3% miss TTFT vs ±2.1%, a -7.3% hit decode
    // vs ±3.9%), outweighed by the hit's TTFT cut. Admission-clamped in the device domain (E7).
    prefix_cache_bytes: PREFIX_CACHE_BYTES,
    // justification: on, MEASURED (cuda-b4-compare.md, cuda-campaign-b4): 40 of 40 on-vs-off cells
    // pass, decode +44–53% on Bonsai and +5% on Qwen3-8B (speculative `off`), TTFT unchanged.
    // Captured steps are token-identical to eager by construction (bit-exact self-check). A step
    // the runner cannot capture runs eager with a named reason. Runtime switch
    // `CANDLE_LLM_CUDA_GRAPHS`; per load `LoadSpec::cuda_graphs`.
    cuda_graphs: true,
    // justification: on — a correctness prerequisite of `cuda_graphs`: without staged positions
    // the dense and Qwen3.5 steps refuse capture (`positions_host_scalar`). MEASURED eager
    // (cuda-b4-compare.md, cuda-campaign-b4, graphs off): decode within -2.3..+10.1% of the
    // host-position path after the `decode_attention` rewrite; the one flagged cell is a +0.8%
    // TTFT against a ±0.4% band. Runtime switch `CANDLE_LLM_DEVICE_POSITIONS`.
    device_positions: true,
    // justification: n/a — Candle's engine has no pipelined loop (CUDA graphs serve that role).
    pipelining: false,
    // justification: on (sc-24133) — the NVRTC device sampler keeps the target distribution; a
    // kernel that fails to compile falls back to the host reference with a named reason.
    device_sampler: true,
    // justification: on (sc-24440) — indexed expert kernels read the routes on the device: no
    // per-layer host read, and the step stays graph-capturable.
    moe_device_dispatch: true,
    // justification: on (sc-24137) — each fused primitive is bit-identical to its op chain.
    // Runtime switch `CANDLE_LLM_FUSED_KERNELS`.
    fused_kernels: true,
    // justification: on (sc-24444) — the fused CUDA rotation is bit-identical to the op chain;
    // it also needs `fused_kernels`.
    fused_rotation: true,
    // justification: n/a — Candle has no fused gated-delta kernel; the recurrence is an op chain.
    gdn_kernel: false,
    // justification: on (sc-24443) — the chunkwise prefill matches the per-token recurrence
    // within its tolerance and replaces `T` sequential steps.
    gdn_chunked_prefill: true,
    // justification: on, MEASURED faster than cuBLASLt for decode-sized NVFP4 projections
    // (sc-24136 evidence) and it serves the indexed NVFP4 MoE kernels. Runtime switch
    // `CANDLE_LLM_NVFP4_GEMV`.
    nvfp4_gemv: true,
    // justification: n/a — an MLX routing.
    sdpa_kernel_routing: false,
};

/// **Candle on Metal** (`candle-llm`, `metal` feature, Metal device).
pub const CANDLE_METAL: DecodeDefaults = DecodeDefaults {
    backend: DecodeBackend::CandleMetal,
    // justification: off — follows CANDLE_CUDA's MEASURED value (Metal was not a campaign host).
    speculative: Speculative::Off,
    // justification: proposer-intrinsic depths.
    recommended_depths: RECOMMENDED_DEPTHS,
    // justification: on — follows CANDLE_CUDA's MEASURED value (Metal was not a campaign host);
    // admission-clamped (E7).
    prefix_cache_bytes: PREFIX_CACHE_BYTES,
    // justification: n/a — CUDA only.
    cuda_graphs: false,
    // justification: off, a correctness limit — Metal has no `write_rows_at` / `decode_attention`
    // kernel, so the device-positions step would fail there; and the path exists to make a step
    // capturable, which only CUDA does.
    device_positions: false,
    // justification: n/a — no pipelined loop in Candle.
    pipelining: false,
    // justification: n/a — the device sampler is a CUDA (NVRTC) kernel; greedy is still the
    // device argmax on every device.
    device_sampler: false,
    // justification: on — a GPU pays a pipeline drain per host route read; the gathered dispatch
    // keeps the routes on the device.
    moe_device_dispatch: true,
    // justification: n/a — the fused primitives are CUDA kernels.
    fused_kernels: false,
    // justification: n/a — the fused rotation is a CUDA kernel.
    fused_rotation: false,
    // justification: n/a — no Candle gated-delta kernel.
    gdn_kernel: false,
    // justification: on (sc-24443), as CUDA.
    gdn_chunked_prefill: true,
    // justification: n/a — NVFP4 is a CUDA format.
    nvfp4_gemv: false,
    // justification: n/a — an MLX routing.
    sdpa_kernel_routing: false,
};

/// **Candle on the CPU** (`candle-llm`, any build, CPU device).
pub const CANDLE_CPU: DecodeDefaults = DecodeDefaults {
    backend: DecodeBackend::CandleCpu,
    // justification: off — follows CANDLE_CUDA's MEASURED value (the CPU was not a campaign host).
    speculative: Speculative::Off,
    // justification: proposer-intrinsic depths.
    recommended_depths: RECOMMENDED_DEPTHS,
    // justification: on — follows CANDLE_CUDA's MEASURED value (the CPU was not a campaign host);
    // admission-clamped in host memory.
    prefix_cache_bytes: PREFIX_CACHE_BYTES,
    // justification: n/a — CUDA only.
    cuda_graphs: false,
    // justification: off — no graph runner on the CPU, and the
    // CPU runs `write_rows_at` / `decode_attention` as host reference code that reads the staged
    // position back every step — an extra upload and read-back per step that buys nothing
    // without a graph to replay.
    device_positions: false,
    // justification: n/a — no pipelined loop in Candle.
    pipelining: false,
    // justification: n/a — the device sampler is a CUDA kernel (greedy is the argmax everywhere).
    device_sampler: false,
    // justification: off, MEASURED — on the CPU a host route read is free and the grouped
    // dispatch reads each routed expert in place, while the gathered dispatch copies them: 1.5–1.8x
    // slower gathered at 64 experts / k = 8 / hidden 1024 / expert FFN 512, t = 1–8
    // (candle-llm `primitives::moe::tests::stacked_vs_grouped_cpu_timing`).
    moe_device_dispatch: false,
    // justification: n/a — the fused primitives are CUDA kernels.
    fused_kernels: false,
    // justification: n/a — the fused rotation is a CUDA kernel.
    fused_rotation: false,
    // justification: n/a — no Candle gated-delta kernel.
    gdn_kernel: false,
    // justification: on (sc-24443), as CUDA.
    gdn_chunked_prefill: true,
    // justification: n/a — NVFP4 is a CUDA format.
    nvfp4_gemv: false,
    // justification: n/a — an MLX routing.
    sdpa_kernel_routing: false,
};

/// The speculative option a request that leaves it unset runs with on `backend` — the value a
/// consumer (e.g. a product's settings UI) shows as "the build's default".
pub const fn speculative_default(backend: DecodeBackend) -> Speculative {
    backend.defaults().speculative
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_row_is_its_backends_row() {
        for backend in DecodeBackend::ALL {
            assert_eq!(backend.defaults().backend, backend, "{}", backend.label());
            assert_eq!(speculative_default(backend), backend.defaults().speculative);
        }
    }

    /// A row may only claim a backend-specific path on the backend that has it.
    #[test]
    fn backend_specific_paths_are_off_where_the_backend_has_none() {
        for row in [MLX, CANDLE_METAL, CANDLE_CPU] {
            assert!(!row.cuda_graphs && !row.nvfp4_gemv, "{:?}", row.backend);
        }
        for row in [CANDLE_CUDA, CANDLE_METAL, CANDLE_CPU] {
            assert!(
                !row.pipelining && !row.gdn_kernel && !row.sdpa_kernel_routing,
                "{:?}",
                row.backend
            );
        }
        for row in [CANDLE_METAL, CANDLE_CPU] {
            assert!(
                !row.device_sampler && !row.fused_kernels && !row.fused_rotation,
                "{:?}",
                row.backend
            );
        }
    }

    #[test]
    fn recommended_depths_are_positive_per_proposer() {
        for backend in DecodeBackend::ALL {
            for proposer in SpeculativeProposer::ALL {
                assert!(backend.defaults().recommended_depths.get(proposer) >= 1);
            }
        }
    }
}
