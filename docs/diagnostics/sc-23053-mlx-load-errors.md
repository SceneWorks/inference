# sc-23053: MiniMax-H3 weight-read errors and embedding validity

MLX fork [PR 29](https://github.com/SceneWorks/mlx-rs/pull/29), merged as
`d5a7fc018d713a37091e1cd102873eab355a00c6`, fixes a demonstrated failure path:
the CPU stream waited on an asynchronous weight-read future without retrieving
its exception. Injecting EFAULT into all 47 reads of the 1,555,824,640-byte bf16
embedding table reproduced an all-zero `[1, 836, 5120]` grounded embedding while
evaluation reported success. With the fix, the same saved input and injected
failure return a recoverable error naming the shard, read offset, byte count,
and errno 14. No generation retry or denoising occurs.

Completion errors now follow array data dependencies through asynchronous
evaluation, graph detachment, repeated host waits, and downstream computations.
A failed array cannot poison independent outputs in the same evaluation batch.
Stream workers still complete their fences; exceptions are raised on the host.
Positioned reads retry EINTR, advance the file offset after partial reads, and
drain outstanding writes before reporting any failure.

The dense H3 embedding boundary checks the requested CPU rows before verifying
the GPU view of the table. Matching CPU/GPU zeros are rejected as invalid data;
valid CPU bytes with a divergent GPU checksum use the existing bounded coherence
verification and retain incidence diagnostics. The final conditioning screen
still rejects zero/non-finite output without retrying generation. Packed token
embeddings keep their existing path and benefit from the shared reader fix.

## Validation on 2026-09-11

All deterministic reader regressions passed on CPU/GPU, synchronous/asynchronous
evaluation: EOF, EFAULT, EINTR, partial reads across the 32 MiB batching boundary,
repeated access, dependent reuse, and independent outputs in the same batch.
Independent source review and targeted clippy passed. All 37 H3 text-encoder unit
tests passed on the final native patch.

Real weights were read from the external model drive using the saved prompt and
image, production keyframe fitting at 768 × 1024, tokenization, vision tower, and
all 50 text layers. Each row covers two text-only and two grounded forwards;
fresh and warm contexts were non-degenerate and exactly equal within each route.
“Fresh” means a newly constructed encoder, not an asserted cold OS file cache.

| Tier | Residency | Text / grounded shapes | Fresh-warm max difference | Observed peak bytes |
| --- | --- | --- | --- | --- |
| bf16 | deferred, two-layer windows | `[1,60,5120]` / `[1,836,5120]` | 0 / 0 | 6,595,342,380 |
| q8 | deferred, two-layer windows | `[1,60,5120]` / `[1,836,5120]` | 0 / 0 | 3,438,556,340 |
| q4 | deferred, two-layer windows | `[1,60,5120]` / `[1,836,5120]` | 0 / 0 | 2,902,645,168 |
| bf16 | resident, lazy initial weights | `[1,60,5120]` / `[1,836,5120]` | 0 / 0 | 54,117,808,684 |

Base model snapshot: `MiniMaxAI/MiniMax-H3@939557dc319dd91227e30195a763f272ba7f8765`.
Packed model snapshot: `SceneWorks/minimax-h3-mlx@137ce668c55a20bc0935fd1cf2a3de8448abb7f4`.
The q8/q4 component configs and embedding headers identify packed U32 tables;
`packed_bits()` alone cannot identify deferred layers, which are not resident.
The committed ignored regression checks token-table packing against the component
config and, in resident mode, checks the actual projection width too.

## Attribution limit

The original failed job did not record the read result or CPU buffer contents.
Its precise mechanism remains unproven: the injected failure establishes a real
loader defect, but cannot retrospectively distinguish that defect from a GPU
visibility failure in the original job. No natural zeroing occurred during these
validation runs. This is a conditioning-stage validation, not a full video render
or an installation receipt for a desktop application.
