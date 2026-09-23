**RTX Pro 6000 / sm_120** — Qwen/Qwen3.8-27B @ 1d4bf0f2ff60 (`bonsai-qwen38-parent`, config sha256 191e0af23210), BF16 greedy, 97 prompt tokens, 256 new tokens per row.

| run | row | tokens | match ref | match baseline ref | tok/s | acceptance | fwd/tok | syncs/tok | device used @ last token | cache live | cache checkpoints |
|---|---|---|---|---|---|---|---|---|---|---|---|
| head-e7805eee4 | MTP off (reference) | 256 | (ref) | yes | 11.23 | n/a | 1.000 | 1.00 | 53.25 GiB | n/a | n/a |
| head-e7805eee4 | MTP off (StepModel, static kv) | 256 | no @107 | no @107 | 11.25 | n/a | 1.000 | 1.00 | 53.38 GiB | 168.9 MiB | 293.6 MiB |

match baseline ref = tokens identical to `head-e7805eee4`'s reference row; device used @ last token = cuMemGetInfo total-free sampled at the row's last generated token while its cache is alive (device-wide, weights included); cache live / checkpoints = the StepModel row's final cache's own accounting (rollback checkpoints separately); syncs/tok = device->host transfers issued by candle-llm per generated token (n/a where the binary predates the counter); fwd/tok = measured target forwards per generated token (n/a where the binary predates the counter).
