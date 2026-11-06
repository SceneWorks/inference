# H3 residency component diagnostic

This ignored CUDA case isolates the first numerical boundaries that differ between
resident and streamed H3 execution. It changes no production math, allocator,
workspace policy, or model settings. Run it only through the explicit
`model-inference-proofs.yml` workflow.

Select
`candle-gen-minimax-h3/candle_gen_minimax_h3/dit::layers::residency_component::production_q4_turbo_workspace_and_qk_parity`.
The default configuration `fixed_cuda` retains physical GPU1. The explicitly
authorized `fixed_cuda_gpu0` configuration selects physical GPU0. Both use logical
CUDA0 under the existing UUID lease and supervisor. No workflow selector changes
or global device defaults are required.

The mandatory manifest defaults to
`D:/Codex/work/sw-f329/envelope-outputs/residency-component-manifest.v1.json`;
`H3_RESIDENCY_COMPONENT_MANIFEST` may override its location, never its accepted
contents. It names the independently frozen accepted staging receipt and the
actual pinned stock q4/Turbo8 assets. Its SHA256
`9df0a1203b8a20d87b81888cb4edeaa34537f50c558aba239a79af5b3c443b01`
is source-bound in the case and registry. Every config, Turbo, and all fourteen
DiT shard lengths/digests must match the accepted receipt's exact repository,
revision, relative path and canonical local path before learned computation.
Missing or altered inputs fail. Outputs go to the supervisor-owned
`MODEL_PROOF_OUTPUT_DIR/residency-component.json`.

The case loads the q4 DiT on CPU and uses the production adapter installer to
resolve canonical Turbo factors. It retains only block0 `to_q`, then compares the
actual GPU-direct constructor with host preparation followed by migration.
Executable frozen factor handles are inspected through a proof-gated accessor;
structured/trainable variants refuse that observation. No full transformer is
materialized on GPU, and no transformer forward is run.

The deterministic BF16 input has the actual `[1,104554,5376]` shape. This is a
component stress input, not a captured activation from the failed full render.
Under the same explicit 32 MiB workspace, runtime packed bytes, executable
factor bytes/scales, base output, isolated residual output, and adapted output
must match exactly. The same raw Q is compared through the original and bounded
weighted RMS and partial RoPE at `[1,104554,56,128]`: 23 chunks of at most 4681
tokens, with a 1572-token tail. Both boundaries require exact BF16 bytes.

The adapted projection also runs with library-default and explicit 12 MiB
workspaces. The report records default-versus-12 MiB, default-versus-32 MiB and
12-versus-32 MiB differences. These are finite-output observations, with no newly
invented tolerance or acceptance waiver. CUDA12.9 documents a 12 MiB default
recommendation for SM12x and 32 MiB for SM10x/SM90; this predeclared contrast tests
a concrete policy confound. See the [cuBLAS workspace documentation](https://docs.nvidia.com/cuda/archive/12.9.0/cublas/index.html#cublassetworkspace).
Each control synchronizes the same stream, resets its default using SetStream,
and retains an explicit buffer through synchronized default restoration. The
proof-pool environment must be absent because its automatic per-GEMM binding
would overwrite these controls. This component case does not prove a 21 GiB cap.

Every comparison covers every element and reports maximum absolute difference,
reference peak, relative peak, F64-accumulated RMS, nonfinite count and original
BF16 SHA256s. Metrics are saved before exact assertions. Successful work requires
six actual projection calls, two residual-only scopes, a positive learned CUDA
device witness, and one late BODY marker. There is no attention, predictor,
denoise, VAE or clip computation. Runtime is unmeasured; provision one bounded
manual case with CPU preparation/readback allowance, not a parameter grid.
