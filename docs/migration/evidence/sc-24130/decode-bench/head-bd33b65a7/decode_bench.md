**RTX Pro 6000 / sm_120** — Qwen/Qwen3.8-27B @ 1d4bf0f2ff60 (`bonsai-qwen38-parent`, config sha256 191e0af23210), BF16 greedy, 97 prompt tokens, 256 new tokens per row.

| run | row | tokens | match ref | match baseline ref | tok/s | acceptance | fwd/tok | syncs/tok | syncs/verify | device used @ last token | cache live | cache checkpoints |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| head-bd33b65a7 | MTP off (reference, growing kv, gqa attn) | 256 | (ref) | yes | 11.21 | n/a | 1.000 | 1.00 | n/a | 53.28 GiB | n/a | n/a |
| head-bd33b65a7 | MTP off (StepModel, static kv, gqa attn) | 256 | yes | yes | 11.37 | n/a | 1.000 | 1.00 | n/a | 53.41 GiB | 168.9 MiB | 293.6 MiB |
| head-bd33b65a7 | MTP K=1 (static kv, gqa attn) | 256 | no @124 | no @124 | 14.18 | 0.827 | 0.645 | 0.55 | 1.00 | 53.41 GiB | n/a | n/a |
| head-bd33b65a7 | MTP K=2 (static kv, gqa attn) | 256 | no @125 | no @125 | 14.60 | 0.668 | 0.613 | 0.43 | 1.00 | 53.41 GiB | n/a | n/a |
| head-bd33b65a7 | MTP K=3 (static kv, gqa attn) | 256 | no @186 | no @186 | 14.53 | 0.640 | 0.535 | 0.35 | 1.00 | 53.41 GiB | n/a | n/a |
| head-bd33b65a7 | MTP K=4 (static kv, gqa attn) | 256 | no @124 | no @124 | 14.67 | 0.549 | 0.535 | 0.32 | 1.00 | 53.41 GiB | n/a | n/a |
| head-bd33b65a7 | MTP K=5 (static kv, gqa attn) | 256 | no @124 | no @124 | 13.66 | 0.480 | 0.547 | 0.30 | 1.00 | 53.41 GiB | n/a | n/a |
| head-bd33b65a7 | n-gram K=3 (static kv, gqa attn) | 256 | no @186 | no @186 | 8.86 | 0.167 | 1.145 | 0.81 | 1.00 | 53.41 GiB | n/a | n/a |

match baseline ref = tokens identical to `head-bd33b65a7`'s reference row; device used @ last token = cuMemGetInfo total-free sampled at the row's last generated token while its cache is alive (device-wide, weights included); cache live / checkpoints = the StepModel row's final cache's own accounting (rollback checkpoints separately); syncs/tok = device->host transfers issued by candle-llm per generated token (n/a where the binary predates the counter); syncs/verify = the speculative engine's transfers per verify step (n/a for non-speculative rows and where the binary predates the engine); fwd/tok = measured target forwards per generated token (n/a where the binary predates the counter).
