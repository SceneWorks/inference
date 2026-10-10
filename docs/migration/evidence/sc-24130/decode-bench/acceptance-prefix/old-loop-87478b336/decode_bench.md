**RTX Pro 6000 / sm_120** — Qwen/Qwen3.8-27B @ 1d4bf0f2ff60 (`bonsai-qwen38-parent`, config sha256 191e0af23210), BF16 greedy, 97 prompt tokens, 124 new tokens per row.

| run | row | tokens | match ref | match baseline ref | tok/s | acceptance | fwd/tok | syncs/tok | syncs/verify | device used @ last token | cache live | cache checkpoints | fused primitives |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| old-loop-87478b336 | MTP off (reference, growing kv, gqa attn) | 124 | (ref) | yes | 11.56 | n/a | 1.000 | 1.00 | n/a | 53.34 GiB | n/a | n/a | n/a |
| old-loop-87478b336 | MTP K=1 (growing kv, gqa attn) | 124 | yes | yes | 14.70 | 0.877 | 0.605 | 1.59 | n/a | 53.34 GiB | n/a | n/a | n/a |
| old-loop-87478b336 | MTP K=2 (growing kv, gqa attn) | 124 | yes | yes | 17.92 | 0.846 | 0.460 | 1.85 | n/a | 53.34 GiB | n/a | n/a | n/a |
| old-loop-87478b336 | MTP K=3 (growing kv, gqa attn) | 124 | yes | yes | 17.40 | 0.709 | 0.468 | 2.22 | n/a | 53.34 GiB | n/a | n/a | n/a |
| old-loop-87478b336 | MTP K=4 (growing kv, gqa attn) | 124 | yes | yes | 17.55 | 0.682 | 0.427 | 2.40 | n/a | 53.34 GiB | n/a | n/a | n/a |
| old-loop-87478b336 | MTP K=5 (growing kv, gqa attn) | 124 | yes | yes | 17.05 | 0.597 | 0.435 | 2.74 | n/a | 53.34 GiB | n/a | n/a | n/a |

match baseline ref = tokens identical to `old-loop-87478b336`'s reference row; device used @ last token = cuMemGetInfo total-free sampled at the row's last generated token while its cache is alive (device-wide, weights included); cache live / checkpoints = the StepModel row's final cache's own accounting (rollback checkpoints separately); syncs/tok = device->host transfers issued by candle-llm per generated token (n/a where the binary predates the counter); syncs/verify = the speculative engine's transfers per verify step (n/a for non-speculative rows and where the binary predates the engine); fwd/tok = measured target forwards per generated token (n/a where the binary predates the counter); fused primitives = the switch the row ran under and how many RMSNorm / SwiGLU / QK-norm+RoPE leaves ran the fused kernel vs the op-chain reference, with the last reference reason (n/a where the binary predates the fused primitives).
