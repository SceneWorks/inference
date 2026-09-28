**RTX Pro 6000 / sm_120** — Qwen/Qwen3.8-27B @ 1d4bf0f2ff60 (`bonsai-qwen38-parent`, config sha256 191e0af23210), BF16 greedy, 97 prompt tokens, 124 new tokens per row.

| run | row | tokens | match ref | match baseline ref | tok/s | acceptance | fwd/tok | syncs/tok | syncs/verify | device used @ last token | cache live | cache checkpoints | fused primitives |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| head-51621c288 | MTP off (reference, growing kv, gqa attn) | 124 | (ref) | yes | 15.73 | n/a | 1.000 | 1.00 | n/a | 53.38 GiB | n/a | n/a | on: 33852 fused / 0 ref |
| head-51621c288 | MTP K=1 (static kv, gqa attn) | 124 | yes | yes | 20.43 | 0.877 | 0.605 | 0.54 | 1.00 | 53.41 GiB | n/a | n/a | on: 21459 fused / 0 ref |
| head-51621c288 | MTP K=2 (static kv, gqa attn) | 124 | yes | yes | 24.24 | 0.846 | 0.460 | 0.38 | 1.00 | 53.41 GiB | n/a | n/a | on: 16633 fused / 0 ref |
| head-51621c288 | MTP K=3 (static kv, gqa attn) | 124 | yes | yes | 22.43 | 0.709 | 0.468 | 0.33 | 1.00 | 53.41 GiB | n/a | n/a | on: 17050 fused / 0 ref |
| head-51621c288 | MTP K=4 (static kv, gqa attn) | 124 | yes | yes | 22.41 | 0.682 | 0.427 | 0.27 | 1.00 | 53.41 GiB | n/a | n/a | on: 15773 fused / 0 ref |
| head-51621c288 | MTP K=5 (static kv, gqa attn) | 124 | yes | yes | 21.50 | 0.597 | 0.435 | 0.26 | 1.00 | 53.41 GiB | n/a | n/a | on: 16206 fused / 0 ref |

match baseline ref = tokens identical to `head-51621c288`'s reference row; device used @ last token = cuMemGetInfo total-free sampled at the row's last generated token while its cache is alive (device-wide, weights included); cache live / checkpoints = the StepModel row's final cache's own accounting (rollback checkpoints separately); syncs/tok = device->host transfers issued by candle-llm per generated token (n/a where the binary predates the counter); syncs/verify = the speculative engine's transfers per verify step (n/a for non-speculative rows and where the binary predates the engine); fwd/tok = measured target forwards per generated token (n/a where the binary predates the counter); fused primitives = the switch the row ran under and how many RMSNorm / SwiGLU / QK-norm+RoPE leaves ran the fused kernel vs the op-chain reference, with the last reference reason (n/a where the binary predates the fused primitives).
