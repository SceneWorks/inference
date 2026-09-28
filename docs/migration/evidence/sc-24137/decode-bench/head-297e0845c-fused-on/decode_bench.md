**RTX Pro 6000 / sm_120** — Qwen/Qwen3.8-27B @ 1d4bf0f2ff60 (`bonsai-qwen38-parent`, config sha256 191e0af23210), BF16 greedy, 97 prompt tokens, 256 new tokens per row.

| run | row | tokens | match ref | match baseline ref | tok/s | acceptance | fwd/tok | syncs/tok | device used @ last token | cache live | cache checkpoints | fused primitives |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| head-297e0845c-fused-on | MTP off (reference) | 256 | (ref) | yes | 14.17 | n/a | 1.000 | 1.00 | 53.38 GiB | n/a | n/a | on: 69888 fused / 0 ref |
| head-297e0845c-fused-on | MTP off (reference, fused off) | 256 | yes | yes | 10.79 | n/a | 1.000 | 1.00 | 53.38 GiB | n/a | n/a | off: 0 fused / 69888 ref (disabled) |
| head-297e0845c-fused-on | MTP off (StepModel) | 256 | yes | yes | 13.76 | n/a | 1.000 | 1.00 | 53.38 GiB | 168.8 MiB | 293.6 MiB | on: 69888 fused / 0 ref |
| head-297e0845c-fused-on | MTP K=1 | 256 | no @107 | no @107 | 15.89 | 0.827 | 0.645 | 1.64 | 53.38 GiB | n/a | n/a | on: 47093 fused / 0 ref |
| head-297e0845c-fused-on | MTP K=2 | 256 | no @107 | no @107 | 18.95 | 0.779 | 0.512 | 1.95 | 53.38 GiB | n/a | n/a | on: 38051 fused / 0 ref |
| head-297e0845c-fused-on | MTP K=3 | 256 | no @107 | no @107 | 16.86 | 0.607 | 0.559 | 2.47 | 53.38 GiB | n/a | n/a | on: 41775 fused / 0 ref |

match baseline ref = tokens identical to `head-297e0845c-fused-on`'s reference row; device used @ last token = cuMemGetInfo total-free sampled at the row's last generated token while its cache is alive (device-wide, weights included); cache live / checkpoints = the StepModel row's final cache's own accounting (rollback checkpoints separately); syncs/tok = device->host transfers issued by candle-llm per generated token (n/a where the binary predates the counter); fwd/tok = measured target forwards per generated token (n/a where the binary predates the counter); fused primitives = the switch the row ran under and how many RMSNorm / SwiGLU / QK-norm+RoPE leaves ran the fused kernel vs the op-chain reference, with the last reference reason (n/a where the binary predates the fused primitives).
