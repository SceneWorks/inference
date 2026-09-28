**RTX Pro 6000 / sm_120** — Qwen/Qwen3.8-27B @ 1d4bf0f2ff60 (`bonsai-qwen38-parent`, config sha256 191e0af23210), BF16 greedy, 97 prompt tokens, 256 new tokens per row.

| run | row | tokens | match ref | match baseline ref | tok/s | acceptance | fwd/tok | syncs/tok | device used @ last token | cache live | cache checkpoints |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline-d2b8cb335 | MTP off (reference) | 256 | (ref) | yes | 11.57 | n/a | n/a | n/a | 53.38 GiB | n/a | n/a |
| baseline-d2b8cb335 | MTP K=1 | 256 | no @107 | no @107 | 14.22 | 0.827 | 0.645 | n/a | 53.38 GiB | n/a | n/a |
| baseline-d2b8cb335 | MTP K=2 | 256 | no @107 | no @107 | 17.11 | 0.779 | 0.512 | n/a | 53.38 GiB | n/a | n/a |
| baseline-d2b8cb335 | MTP K=3 | 256 | no @107 | no @107 | 14.37 | 0.607 | 0.559 | n/a | 53.38 GiB | n/a | n/a |
| baseline-d2b8cb335 | MTP K=4 | 256 | no @107 | no @107 | 13.67 | 0.493 | 0.609 | n/a | 53.38 GiB | n/a | n/a |
| baseline-d2b8cb335 | MTP K=5 | 256 | no @107 | no @107 | 14.17 | 0.480 | 0.547 | n/a | 53.38 GiB | n/a | n/a |
| head-4a059a87b | MTP off (reference) | 256 | (ref) | yes | 12.17 | n/a | 1.000 | 1.00 | 53.31 GiB | n/a | n/a |
| head-4a059a87b | MTP off (StepModel) | 256 | yes | yes | 11.85 | n/a | 1.000 | 1.00 | 53.41 GiB | 168.8 MiB | 293.6 MiB |
| head-4a059a87b | MTP K=1 | 256 | no @107 | no @107 | 14.99 | 0.827 | 0.645 | 1.64 | 53.31 GiB | n/a | n/a |
| head-4a059a87b | MTP K=2 | 256 | no @107 | no @107 | 16.67 | 0.779 | 0.512 | 1.95 | 53.31 GiB | n/a | n/a |
| head-4a059a87b | MTP K=3 | 256 | no @107 | no @107 | 14.85 | 0.607 | 0.559 | 2.47 | 53.31 GiB | n/a | n/a |
| head-4a059a87b | MTP K=4 | 256 | no @107 | no @107 | 13.29 | 0.493 | 0.609 | 3.01 | 53.31 GiB | n/a | n/a |
| head-4a059a87b | MTP K=5 | 256 | no @107 | no @107 | 13.39 | 0.480 | 0.547 | 3.21 | 53.31 GiB | n/a | n/a |
| head-e7805eee4 | MTP off (reference) | 256 | (ref) | yes | 11.23 | n/a | 1.000 | 1.00 | 53.25 GiB | n/a | n/a |
| head-e7805eee4 | MTP off (StepModel, static kv) | 256 | no @107 | no @107 | 11.25 | n/a | 1.000 | 1.00 | 53.38 GiB | 168.9 MiB | 293.6 MiB |
| head-adce2370e-growing-kv | MTP off (reference) | 256 | (ref) | yes | 10.17 | n/a | 1.000 | 1.00 | 53.25 GiB | n/a | n/a |
| head-adce2370e-growing-kv | MTP off (StepModel, growing kv) | 256 | yes | yes | 10.12 | n/a | 1.000 | 1.00 | 53.47 GiB | 168.8 MiB | 293.6 MiB |
| head-23905fe60 | MTP off (reference, growing kv, gqa attn) | 256 | (ref) | no @107 | 11.27 | n/a | 1.000 | 1.00 | 53.28 GiB | n/a | n/a |
| head-23905fe60 | MTP off (StepModel, static kv, gqa attn) | 256 | yes | no @107 | 11.38 | n/a | 1.000 | 1.00 | 53.41 GiB | 168.9 MiB | 293.6 MiB |
| head-978307d88-expanded-attn | MTP off (reference, growing kv, expanded attn) | 256 | (ref) | yes | 9.29 | n/a | 1.000 | 1.00 | 53.31 GiB | n/a | n/a |
| head-978307d88-expanded-attn | MTP off (StepModel, static kv, gqa attn) | 256 | no @107 | no @107 | 9.99 | n/a | 1.000 | 1.00 | 53.41 GiB | 168.9 MiB | 293.6 MiB |

match baseline ref = tokens identical to `baseline-d2b8cb335`'s reference row; device used @ last token = cuMemGetInfo total-free sampled at the row's last generated token while its cache is alive (device-wide, weights included); cache live / checkpoints = the StepModel row's final cache's own accounting (rollback checkpoints separately); syncs/tok = device->host transfers issued by candle-llm per generated token (n/a where the binary predates the counter); fwd/tok = measured target forwards per generated token (n/a where the binary predates the counter).
