**RTX Pro 6000 / sm_120** — Qwen/Qwen3.8-27B @ 1d4bf0f2ff60 (`bonsai-qwen38-parent`, config sha256 191e0af23210), BF16 greedy, 97 prompt tokens, 256 new tokens per row.

| run | row | tokens | match ref | match baseline ref | tok/s | acceptance | fwd/tok | syncs/tok | syncs/verify | device used @ last token | cache live | cache checkpoints |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline-d2b8cb335 | MTP off (reference) | 256 | (ref) | yes | 11.57 | n/a | n/a | n/a | n/a | 53.38 GiB | n/a | n/a |
| baseline-d2b8cb335 | MTP K=1 | 256 | no @107 | no @107 | 14.22 | 0.827 | 0.645 | n/a | n/a | 53.38 GiB | n/a | n/a |
| baseline-d2b8cb335 | MTP K=2 | 256 | no @107 | no @107 | 17.11 | 0.779 | 0.512 | n/a | n/a | 53.38 GiB | n/a | n/a |
| baseline-d2b8cb335 | MTP K=3 | 256 | no @107 | no @107 | 14.37 | 0.607 | 0.559 | n/a | n/a | 53.38 GiB | n/a | n/a |
| baseline-d2b8cb335 | MTP K=4 | 256 | no @107 | no @107 | 13.67 | 0.493 | 0.609 | n/a | n/a | 53.38 GiB | n/a | n/a |
| baseline-d2b8cb335 | MTP K=5 | 256 | no @107 | no @107 | 14.17 | 0.480 | 0.547 | n/a | n/a | 53.38 GiB | n/a | n/a |
| head-4a059a87b | MTP off (reference) | 256 | (ref) | yes | 12.17 | n/a | 1.000 | 1.00 | n/a | 53.31 GiB | n/a | n/a |
| head-4a059a87b | MTP off (StepModel) | 256 | yes | yes | 11.85 | n/a | 1.000 | 1.00 | n/a | 53.41 GiB | 168.8 MiB | 293.6 MiB |
| head-4a059a87b | MTP K=1 | 256 | no @107 | no @107 | 14.99 | 0.827 | 0.645 | 1.64 | n/a | 53.31 GiB | n/a | n/a |
| head-4a059a87b | MTP K=2 | 256 | no @107 | no @107 | 16.67 | 0.779 | 0.512 | 1.95 | n/a | 53.31 GiB | n/a | n/a |
| head-4a059a87b | MTP K=3 | 256 | no @107 | no @107 | 14.85 | 0.607 | 0.559 | 2.47 | n/a | 53.31 GiB | n/a | n/a |
| head-4a059a87b | MTP K=4 | 256 | no @107 | no @107 | 13.29 | 0.493 | 0.609 | 3.01 | n/a | 53.31 GiB | n/a | n/a |
| head-4a059a87b | MTP K=5 | 256 | no @107 | no @107 | 13.39 | 0.480 | 0.547 | 3.21 | n/a | 53.31 GiB | n/a | n/a |
| head-23905fe60 | MTP off (reference, growing kv, gqa attn) | 256 | (ref) | no @107 | 11.27 | n/a | 1.000 | 1.00 | n/a | 53.28 GiB | n/a | n/a |
| head-23905fe60 | MTP off (StepModel, static kv, gqa attn) | 256 | yes | no @107 | 11.38 | n/a | 1.000 | 1.00 | n/a | 53.41 GiB | 168.9 MiB | 293.6 MiB |
| head-bd33b65a7 | MTP off (reference, growing kv, gqa attn) | 256 | (ref) | no @107 | 11.21 | n/a | 1.000 | 1.00 | n/a | 53.28 GiB | n/a | n/a |
| head-bd33b65a7 | MTP off (StepModel, static kv, gqa attn) | 256 | yes | no @107 | 11.37 | n/a | 1.000 | 1.00 | n/a | 53.41 GiB | 168.9 MiB | 293.6 MiB |
| head-bd33b65a7 | MTP K=1 (static kv, gqa attn) | 256 | no @124 | no @107 | 14.18 | 0.827 | 0.645 | 0.55 | 1.00 | 53.41 GiB | n/a | n/a |
| head-bd33b65a7 | MTP K=2 (static kv, gqa attn) | 256 | no @125 | no @107 | 14.60 | 0.668 | 0.613 | 0.43 | 1.00 | 53.41 GiB | n/a | n/a |
| head-bd33b65a7 | MTP K=3 (static kv, gqa attn) | 256 | no @186 | no @107 | 14.53 | 0.640 | 0.535 | 0.35 | 1.00 | 53.41 GiB | n/a | n/a |
| head-bd33b65a7 | MTP K=4 (static kv, gqa attn) | 256 | no @124 | no @107 | 14.67 | 0.549 | 0.535 | 0.32 | 1.00 | 53.41 GiB | n/a | n/a |
| head-bd33b65a7 | MTP K=5 (static kv, gqa attn) | 256 | no @124 | no @107 | 13.66 | 0.480 | 0.547 | 0.30 | 1.00 | 53.41 GiB | n/a | n/a |
| head-bd33b65a7 | n-gram K=3 (static kv, gqa attn) | 256 | no @186 | no @107 | 8.86 | 0.167 | 1.145 | 0.81 | 1.00 | 53.41 GiB | n/a | n/a |

match baseline ref = tokens identical to `baseline-d2b8cb335`'s reference row; device used @ last token = cuMemGetInfo total-free sampled at the row's last generated token while its cache is alive (device-wide, weights included); cache live / checkpoints = the StepModel row's final cache's own accounting (rollback checkpoints separately); syncs/tok = device->host transfers issued by candle-llm per generated token (n/a where the binary predates the counter); syncs/verify = the speculative engine's transfers per verify step (n/a for non-speculative rows and where the binary predates the engine); fwd/tok = measured target forwards per generated token (n/a where the binary predates the counter).
