# Speculative paths (SW-7200)

Every ratio is against the `off` row of the same process. `draft ms/pass` is the proposer's host wall time per call: where the proposal comes back on the host it waited for the draft's device work and anything still queued ahead of it (the previous verify), so it is an upper bound on the draft's own cost; rows marked `enqueue-only` left their drafts on the device (Candle greedy MTP / draft model), so it is a lower bound. Peak GiB is the backend's reading (MLX allocator high-water; CUDA device-used at tokens, a lower bound) and `+request` what the request added over its start.

- refused: decode-speedups-bench-mlx-9f3250252ef07ed718ba04c447dd80be014d7b60-44587316639-1/q4-dspine-cold epic-1: exited 241
- refused: decode-speedups-bench-mlx-9f3250252ef07ed718ba04c447dd80be014d7b60-44587316639-1/q4-dspine-cold: 0 epic processes, the row declares 1
- refused: decode-speedups-bench-mlx-9f3250252ef07ed718ba04c447dd80be014d7b60-44587316639-1/q4-dflash2-cold epic-1: exited 241
- refused: decode-speedups-bench-mlx-9f3250252ef07ed718ba04c447dd80be014d7b60-44587316639-1/q4-dflash2-cold: 0 epic processes, the row declares 1

## candle-cuda · qwen3.8-27b-abliterated · q4 · cold context · 16 tokens · thinking off

| path | ran | decode tok/s | ×off decode (pred / open) | full request ms | ×off request | TTFT ms | accepted/pass | draft ms/pass (share) | peak GiB (+request) | fallbacks |
|---|---|---|---|---|---|---|---|---|---|---|
| draft_model | none | 47.6 | 1.02 (1.04 / 1.00) | 552 | 1.08 | 215 | — | — | 20.5 (+0.03) | speculative: `draft_model` is not available for this model (this model does not advertise it; decoded without a proposer) |
| off | none | 46.6 | 1.00 (1.00 / 1.00) | 597 | 1.00 | 252 | — | — | 20.5 (+0.16) | draft model: its tokenizer vocabulary is not the target's (vocabulary sizes differ (248077 vs 248070 tokens)) |

## candle-cuda · qwen3.8-27b-abliterated · q4 · cold context · 256 tokens · thinking off

| path | ran | decode tok/s | ×off decode (pred / open) | full request ms | ×off request | TTFT ms | accepted/pass | draft ms/pass (share) | peak GiB (+request) | fallbacks |
|---|---|---|---|---|---|---|---|---|---|---|
| block_draft:dflash2:linear | block_draft | 94.2 | 1.46 (2.10 / 0.85) | 2925 | 1.38 | 279 | 4.41 | 12.37 (23%) | 23.5 (+0.03) |  |
| block_draft:dflash2:linear@recommended | block_draft | 90.9 | 1.49 (1.87 / 1.06) | 2602 | 1.40 | 279 | 2.83 | 12.95 (29%) | 23.0 (+0.03) |  |
| block_draft:dflash2:tree | block_draft | 28.5 | 0.45 (0.64 / 0.27) | 9075 | 0.45 | 276 | 4.73 (tree 4.73) | 11.40 (6%) | 26.6 (+2.97) |  |
| block_draft:dspine:linear | block_draft | 93.0 | 1.41 (2.03 / 0.82) | 3011 | 1.34 | 251 | 4.15 | 11.38 (21%) | 23.9 (+0.03) |  |
| block_draft:dspine:linear@recommended | block_draft | 91.5 | 1.48 (1.87 / 1.04) | 2598 | 1.42 | 237 | 2.69 | 10.10 (26%) | 23.4 (+0.03) |  |
| block_draft:dspine:tree | block_draft | 33.1 | 0.50 (0.73 / 0.29) | 8119 | 0.52 | 260 | 4.53 (tree 4.53) | 10.13 (6%) | 27.1 (+3.16) |  |
| mtp | mtp | 91.7 | 1.40 (1.96 / 0.85) | 2920 | 1.35 | 228 | 3.89 | 2.12 (4%) enqueue-only | 23.9 (+0.03) |  |
| mtp@recommended | mtp | 102.6 | 1.70 (1.98 / 1.35) | 2164 | 1.61 | 231 | 2.19 | 1.03 (3%) enqueue-only | 23.1 (+0.03) |  |
| off | none | 58.7 | 1.00 (1.00 / 1.00) | 3279 | 1.00 | 223 | — | — | 22.7 (+0.03) |  |
| prompt_lookup | prompt_lookup | 65.3 | 0.98 (1.32 / 0.63) | 3943 | 0.97 | 241 | 1.45 | 0.00 (0%) | 23.9 (+0.03) |  |

## candle-cuda · qwen3.8-27b-abliterated · q8 · cold context · 256 tokens · thinking off

| path | ran | decode tok/s | ×off decode (pred / open) | full request ms | ×off request | TTFT ms | accepted/pass | draft ms/pass (share) | peak GiB (+request) | fallbacks |
|---|---|---|---|---|---|---|---|---|---|---|
| block_draft:dflash2:linear | block_draft | 89.5 | 2.12 (2.96 / 1.29) | 2973 | 1.95 | 287 | 4.41 | 12.63 (23%) | 35.7 (+0.03) |  |
| block_draft:dflash2:linear@recommended | block_draft | 80.5 | 2.00 (2.46 / 1.47) | 2824 | 1.88 | 263 | 2.80 | 11.94 (25%) | 35.1 (+0.03) |  |
| block_draft:dflash2:tree | block_draft | 26.2 | 0.64 (0.82 / 0.45) | 8567 | 0.65 | 244 | 4.83 (tree 4.83) | 12.76 (6%) | 38.7 (+2.94) |  |
| block_draft:dspine:linear | block_draft | 87.7 | 2.08 (2.99 / 1.21) | 3116 | 1.94 | 268 | 4.24 | 11.40 (20%) | 36.0 (+0.03) |  |
| block_draft:dspine:linear@recommended | block_draft | 76.8 | 1.92 (2.43 / 1.36) | 2993 | 1.83 | 237 | 2.70 | 11.96 (24%) | 35.4 (+0.03) |  |
| block_draft:dspine:tree | block_draft | 28.9 | 0.70 (0.99 / 0.41) | 8808 | 0.70 | 261 | 4.66 (tree 4.66) | 11.37 (6%) | 39.0 (+2.94) |  |
| mtp | mtp | 85.4 | 2.08 (2.83 / 1.31) | 3078 | 1.88 | 360 | 4.31 | 2.54 (4%) enqueue-only | 36.0 (+0.03) |  |
| mtp@recommended | mtp | 79.1 | 2.04 (2.33 / 1.67) | 2698 | 1.93 | 264 | 2.25 | 1.06 (3%) enqueue-only | 35.2 (+0.03) |  |
| off | none | 38.4 | 1.00 (1.00 / 1.00) | 4892 | 1.00 | 255 | — | — | 34.8 (+0.03) |  |
| prompt_lookup | prompt_lookup | 54.6 | 1.25 (1.68 / 0.81) | 4698 | 1.22 | 282 | 1.30 | 0.00 (0%) | 36.0 (+0.03) |  |

## candle-cuda · qwen3.8-27b-abliterated · q4 · cold context · 64 tokens · thinking off

| path | ran | decode tok/s | ×off decode (pred / open) | full request ms | ×off request | TTFT ms | accepted/pass | draft ms/pass (share) | peak GiB (+request) | fallbacks |
|---|---|---|---|---|---|---|---|---|---|---|
| mtp@recommended | mtp | 93.3 | 1.61 (1.84 / 1.31) | 949 | 1.45 | 238 | 2.09 | 1.00 (3%) enqueue-only | 23.9 (+0.22) |  |
| off | none | 54.7 | 1.00 (1.00 / 1.00) | 1468 | 1.00 | 281 | — | — | 23.4 (+0.22) |  |

## candle-cuda · qwen3.8-27b-abliterated · q8 · cold context · 64 tokens · thinking off

| path | ran | decode tok/s | ×off decode (pred / open) | full request ms | ×off request | TTFT ms | accepted/pass | draft ms/pass (share) | peak GiB (+request) | fallbacks |
|---|---|---|---|---|---|---|---|---|---|---|
| mtp@recommended | mtp | 75.3 | 1.96 (2.19 / 1.67) | 1169 | 1.71 | 299 | 2.29 | 1.05 (2%) enqueue-only | 35.9 (+0.16) |  |
| off | none | 37.9 | 1.00 (1.00 / 1.00) | 1933 | 1.00 | 251 | — | — | 35.4 (+0.25) |  |

## candle-cuda · qwen3.8-27b-abliterated · q4 · reused context · 64 tokens · thinking off

| path | ran | decode tok/s | ×off decode (pred / open) | full request ms | ×off request | TTFT ms | accepted/pass | draft ms/pass (share) | peak GiB (+request) | fallbacks |
|---|---|---|---|---|---|---|---|---|---|---|
| block_draft:dflash2:linear | block_draft | 59.6 | 1.11 (1.34 / 0.84) | 1298 | 1.08 | 135 | 2.84 | 13.18 (20%) | 24.4 (+0.03) |  |
| block_draft:dflash2:linear@recommended | block_draft | 67.5 | 1.26 (1.41 / 1.05) | 1170 | 1.22 | 132 | 2.33 | 16.09 (26%) | 23.8 (+0.03) |  |
| block_draft:dflash2:tree | block_draft | 23.7 | 0.43 (0.50 / 0.34) | 3313 | 0.44 | 132 | 3.54 (tree 3.60) | 14.81 (7%) | 27.8 (+3.41) |  |
| block_draft:dspine:linear | block_draft | 66.8 | 1.13 (1.40 / 0.82) | 1146 | 1.09 | 104 | 2.87 | 11.47 (20%) | 24.5 (+0.03) |  |
| block_draft:dspine:linear@recommended | block_draft | 76.5 | 1.32 (1.54 / 1.05) | 974 | 1.26 | 100 | 2.30 | 10.29 (24%) | 24.0 (+0.03) |  |
| block_draft:dspine:tree | block_draft | 20.3 | 0.34 (0.40 / 0.27) | 3665 | 0.34 | 110 | 3.36 (tree 3.38) | 13.73 (6%) | 27.9 (+3.38) |  |
| mtp | mtp | 85.7 | 1.41 (1.92 / 0.88) | 976 | 1.32 | 105 | 4.07 | 1.95 (3%) enqueue-only | 24.5 (+0.03) |  |
| mtp@recommended | mtp | 92.9 | 1.61 (1.85 / 1.31) | 809 | 1.51 | 93 | 2.09 | 1.00 (3%) enqueue-only | 23.9 (+0.00) |  |
| off | none | 54.3 | 1.00 (1.00 / 1.00) | 1293 | 1.00 | 99 | — | — | 23.4 (+0.03) |  |
| prompt_lookup | prompt_lookup | 58.1 | 0.95 (1.20 / 0.67) | 1384 | 0.92 | 107 | 1.27 | 0.00 (0%) | 24.5 (+0.05) |  |

## candle-cuda · qwen3.8-27b-abliterated · q8 · reused context · 64 tokens · thinking off

| path | ran | decode tok/s | ×off decode (pred / open) | full request ms | ×off request | TTFT ms | accepted/pass | draft ms/pass (share) | peak GiB (+request) | fallbacks |
|---|---|---|---|---|---|---|---|---|---|---|
| block_draft:dflash2:linear | block_draft | 65.5 | 1.67 (1.97 / 1.30) | 1153 | 1.60 | 110 | 2.99 | 12.49 (21%) | 36.4 (+0.03) |  |
| block_draft:dflash2:linear@recommended | block_draft | 70.0 | 1.82 (2.03 / 1.55) | 1038 | 1.74 | 99 | 2.52 | 11.85 (22%) | 36.0 (+0.03) |  |
| block_draft:dflash2:tree | block_draft | 25.1 | 0.65 (0.70 / 0.57) | 2774 | 0.66 | 112 | 3.43 (tree 3.43) | 10.86 (6%) | 39.8 (+3.44) |  |
| block_draft:dspine:linear | block_draft | 63.8 | 1.62 (2.02 / 1.16) | 1320 | 1.46 | 211 | 2.95 | 11.98 (20%) | 36.6 (+0.03) |  |
| block_draft:dspine:linear@recommended | block_draft | 66.1 | 1.70 (2.07 / 1.27) | 1193 | 1.58 | 151 | 2.37 | 11.91 (23%) | 36.2 (+0.03) |  |
| block_draft:dspine:tree | block_draft | 22.7 | 0.56 (0.76 / 0.35) | 3506 | 0.56 | 191 | 3.48 (tree 3.54) | 12.19 (6%) | 39.8 (+3.38) |  |
| mtp | mtp | 79.0 | 1.99 (2.59 / 1.35) | 1071 | 1.79 | 161 | 4.25 | 2.31 (4%) enqueue-only | 36.6 (+0.03) |  |
| mtp@recommended | mtp | 75.8 | 2.00 (2.21 / 1.72) | 987 | 1.85 | 125 | 2.29 | 1.05 (2%) enqueue-only | 35.9 (+0.00) |  |
| off | none | 37.7 | 1.00 (1.00 / 1.00) | 1793 | 1.00 | 101 | — | — | 35.4 (+0.03) |  |
| prompt_lookup | prompt_lookup | 49.5 | 1.20 (1.56 / 0.80) | 1698 | 1.15 | 147 | 1.19 | 0.00 (0%) | 36.6 (+0.04) |  |

## mlx · qwen3.8-27b-abliterated · q4 · cold context · 16 tokens · thinking off

| path | ran | decode tok/s | ×off decode (pred / open) | full request ms | ×off request | TTFT ms | accepted/pass | draft ms/pass (share) | peak GiB (+request) | fallbacks |
|---|---|---|---|---|---|---|---|---|---|---|
| draft_model | none | 30.1 | 1.00 (1.00 / 1.00) | 760 | 1.01 | 229 | — | — | 19.3 (+0.72) | speculative: `draft_model` is not available for this model (this model does not advertise it; decoded without a proposer) |
| off | none | 30.2 | 1.00 (1.00 / 1.00) | 771 | 1.00 | 241 | — | — | 19.3 (+0.72) | draft model: its tokenizer vocabulary is not the target's (vocabulary sizes differ (248077 vs 248070 tokens)) |

## mlx · qwen3.8-27b-abliterated · q4 · cold context · 256 tokens · thinking off

| path | ran | decode tok/s | ×off decode (pred / open) | full request ms | ×off request | TTFT ms | accepted/pass | draft ms/pass (share) | peak GiB (+request) | fallbacks |
|---|---|---|---|---|---|---|---|---|---|---|
| block_draft:dflash2:linear | block_draft | 46.9 | 1.63 (2.19 / 1.05) | 5099 | 1.55 | 251 | 2.76 | 13.92 (17%) | 21.7 (+1.09) |  |
| block_draft:dflash2:linear@recommended | block_draft | 47.3 | 1.64 (2.21 / 1.05) | 5083 | 1.56 | 247 | 2.76 | 13.83 (17%) | 21.7 (+1.09) |  |
| block_draft:dflash2:tree | block_draft | 24.3 | 0.88 (1.04 / 0.68) | 8504 | 0.88 | 256 | 3.25 (tree 3.26) | 13.90 (8%) | 24.8 (+4.18) |  |
| block_draft:dspine:linear | block_draft | 46.7 | 1.59 (2.22 / 0.97) | 5383 | 1.51 | 250 | 2.63 | 13.31 (17%) | 22.0 (+1.12) |  |
| block_draft:dspine:linear@recommended | block_draft | 46.9 | 1.60 (2.23 / 0.97) | 5371 | 1.52 | 248 | 2.63 | 13.27 (17%) | 22.0 (+1.12) |  |
| block_draft:dspine:tree | block_draft | 23.3 | 0.83 (1.04 / 0.58) | 9409 | 0.82 | 250 | 2.98 (tree 2.98) | 13.16 (8%) | 25.1 (+4.17) |  |
| mtp | mtp | 35.3 | 1.25 (1.58 / 0.88) | 6274 | 1.22 | 236 | 2.77 | 25.05 (24%) | 21.9 (+1.05) |  |
| mtp@recommended | mtp | 34.3 | 1.24 (1.46 / 0.96) | 6054 | 1.22 | 235 | 2.17 | 18.99 (21%) | 21.8 (+0.90) |  |
| off | none | 27.2 | 1.00 (1.00 / 1.00) | 6855 | 1.00 | 233 | — | — | 21.6 (+0.72) |  |
| prompt_lookup | prompt_lookup | 36.4 | 1.23 (1.66 / 0.79) | 6616 | 1.20 | 245 | 1.10 | 0.00 (0%) | 21.9 (+1.04) |  |

## mlx · qwen3.8-27b-abliterated · q8 · cold context · 256 tokens · thinking off

| path | ran | decode tok/s | ×off decode (pred / open) | full request ms | ×off request | TTFT ms | accepted/pass | draft ms/pass (share) | peak GiB (+request) | fallbacks |
|---|---|---|---|---|---|---|---|---|---|---|
| block_draft:dflash2:linear | block_draft | 33.8 | 2.14 (2.72 / 1.49) | 6743 | 2.05 | 351 | 2.75 | 17.17 (15%) | 33.2 (+1.09) |  |
| block_draft:dflash2:linear@recommended | block_draft | 28.4 | 1.81 (2.17 / 1.37) | 9161 | 1.56 | 406 | 2.75 | 33.07 (15%) | 33.2 (+1.09) |  |
| block_draft:dflash2:tree | block_draft | 18.2 | 1.18 (1.35 / 0.96) | 11401 | 1.17 | 335 | 3.31 (tree 3.31) | 19.81 (7%) | 36.3 (+4.18) |  |
| block_draft:dspine:linear | block_draft | 28.1 | 1.63 (1.94 / 1.25) | 7452 | 1.57 | 340 | 2.60 | 18.65 (14%) | 33.5 (+1.12) |  |
| block_draft:dspine:linear@recommended | block_draft | 29.9 | 1.75 (2.23 / 1.22) | 7622 | 1.66 | 391 | 2.60 | 16.51 (15%) | 33.5 (+1.12) |  |
| block_draft:dspine:tree | block_draft | 18.1 | 1.09 (1.29 / 0.84) | 11218 | 1.08 | 342 | 3.10 (tree 3.11) | 14.79 (7%) | 36.6 (+4.17) |  |
| mtp | mtp | 23.5 | 1.40 (1.58 / 1.18) | 9231 | 1.18 | 343 | 2.74 | 46.04 (18%) | 33.5 (+1.04) |  |
| mtp@recommended | mtp | 25.9 | 1.56 (1.84 / 1.22) | 7828 | 1.51 | 336 | 2.18 | 19.89 (16%) | 33.3 (+0.90) |  |
| off | none | 15.8 | 1.00 (1.00 / 1.00) | 11874 | 1.00 | 324 | — | — | 33.0 (+0.62) |  |
| prompt_lookup | prompt_lookup | 24.3 | 1.36 (1.82 / 0.87) | 9621 | 1.33 | 344 | 1.04 | 0.00 (0%) | 33.5 (+1.04) |  |

## mlx · qwen3.8-27b-abliterated · q4 · cold context · 64 tokens · thinking off

| path | ran | decode tok/s | ×off decode (pred / open) | full request ms | ×off request | TTFT ms | accepted/pass | draft ms/pass (share) | peak GiB (+request) | fallbacks |
|---|---|---|---|---|---|---|---|---|---|---|
| mtp@recommended | mtp | 31.5 | 1.10 (1.24 / 0.92) | 2455 | 1.06 | 296 | 2.17 | 18.80 (18%) | 22.7 (+1.06) |  |
| off | none | 27.8 | 1.00 (1.00 / 1.00) | 2553 | 1.00 | 245 | — | — | 22.2 (+0.93) |  |

## mlx · qwen3.8-27b-abliterated · q8 · cold context · 64 tokens · thinking off

| path | ran | decode tok/s | ×off decode (pred / open) | full request ms | ×off request | TTFT ms | accepted/pass | draft ms/pass (share) | peak GiB (+request) | fallbacks |
|---|---|---|---|---|---|---|---|---|---|---|
| mtp@recommended | mtp | 29.6 | 1.69 (1.90 / 1.41) | 2490 | 1.60 | 280 | 2.28 | 18.09 (16%) | 34.3 (+1.06) |  |
| off | none | 17.4 | 1.00 (1.00 / 1.00) | 3948 | 1.00 | 273 | — | — | 33.7 (+0.81) |  |

## mlx · qwen3.8-27b-abliterated · q4 · reused context · 64 tokens · thinking off

| path | ran | decode tok/s | ×off decode (pred / open) | full request ms | ×off request | TTFT ms | accepted/pass | draft ms/pass (share) | peak GiB (+request) | fallbacks |
|---|---|---|---|---|---|---|---|---|---|---|
| block_draft:dflash2:linear | block_draft | 42.5 | 2.05 (1.81 / 2.46) | 1734 | 1.99 | 89 | 2.38 | 14.29 (17%) | 22.5 (+1.07) |  |
| block_draft:dflash2:linear@recommended | block_draft | 42.0 | 1.97 (1.84 / 2.19) | 1862 | 1.92 | 96 | 2.38 | 14.58 (17%) | 22.5 (+1.08) |  |
| block_draft:dflash2:tree | block_draft | 21.3 | 1.00 (0.92 / 1.16) | 3701 | 0.99 | 88 | 3.00 (tree 3.03) | 21.02 (8%) | 25.6 (+4.18) |  |
| block_draft:dspine:linear | block_draft | 36.7 | 1.25 (1.71 / 0.78) | 2185 | 1.23 | 93 | 1.99 | 14.39 (16%) | 22.8 (+1.10) |  |
| block_draft:dspine:linear@recommended | block_draft | 35.5 | 1.15 (1.70 / 0.64) | 2938 | 1.07 | 95 | 1.99 | 14.52 (16%) | 22.8 (+1.10) |  |
| block_draft:dspine:tree | block_draft | 18.8 | 0.67 (0.82 / 0.50) | 3797 | 0.68 | 94 | 2.48 (tree 2.50) | 14.07 (7%) | 25.7 (+4.02) |  |
| mtp | mtp | 30.2 | 1.01 (1.40 / 0.62) | 2828 | 0.99 | 103 | 2.66 | 34.51 (22%) | 22.7 (+1.05) |  |
| mtp@recommended | mtp | 30.2 | 1.07 (1.16 / 0.96) | 2441 | 1.06 | 116 | 2.17 | 18.80 (19%) | 22.6 (+0.90) |  |
| off | none | 25.1 | 1.00 (1.00 / 1.00) | 3433 | 1.00 | 86 | — | — | 21.9 (+0.20) |  |
| prompt_lookup | prompt_lookup | 30.8 | 0.99 (1.48 / 0.54) | 2964 | 0.98 | 95 | 0.94 | 0.00 (0%) | 22.7 (+1.04) |  |

## mlx · qwen3.8-27b-abliterated · q8 · reused context · 64 tokens · thinking off

| path | ran | decode tok/s | ×off decode (pred / open) | full request ms | ×off request | TTFT ms | accepted/pass | draft ms/pass (share) | peak GiB (+request) | fallbacks |
|---|---|---|---|---|---|---|---|---|---|---|
| block_draft:dflash2:linear | block_draft | 36.6 | 2.03 (2.36 / 1.63) | 1976 | 1.98 | 99 | 2.48 | 14.49 (15%) | 34.0 (+1.07) |  |
| block_draft:dflash2:linear@recommended | block_draft | 37.0 | 2.06 (2.39 / 1.64) | 1954 | 2.00 | 97 | 2.48 | 14.27 (15%) | 34.0 (+1.07) |  |
| block_draft:dflash2:tree | block_draft | 20.4 | 1.16 (1.29 / 0.98) | 3325 | 1.15 | 100 | 3.02 (tree 3.03) | 14.30 (7%) | 37.1 (+4.15) |  |
| block_draft:dspine:linear | block_draft | 33.8 | 1.84 (2.38 / 1.24) | 2253 | 1.79 | 103 | 2.26 | 13.97 (14%) | 34.3 (+1.10) |  |
| block_draft:dspine:linear@recommended | block_draft | 33.6 | 1.83 (2.38 / 1.23) | 2253 | 1.78 | 101 | 2.26 | 14.32 (14%) | 34.3 (+1.10) |  |
| block_draft:dspine:tree | block_draft | 17.5 | 0.95 (1.24 / 0.65) | 4286 | 0.95 | 105 | 2.60 (tree 2.61) | 15.17 (7%) | 37.2 (+4.01) |  |
| mtp | mtp | 32.0 | 1.80 (2.17 / 1.37) | 2199 | 1.76 | 99 | 2.88 | 23.81 (19%) | 34.2 (+1.04) |  |
| mtp@recommended | mtp | 30.1 | 1.72 (1.95 / 1.42) | 2277 | 1.68 | 99 | 2.28 | 18.09 (16%) | 34.1 (+0.90) |  |
| off | none | 17.3 | 1.00 (1.00 / 1.00) | 3775 | 1.00 | 96 | — | — | 33.4 (+0.19) |  |
| prompt_lookup | prompt_lookup | 26.0 | 1.39 (1.81 / 0.93) | 2952 | 1.37 | 95 | 0.94 | 0.00 (0%) | 34.2 (+1.04) |  |

## Block-draft inputs for `auto`

Each block draft at its recommended depth, linear, against `mtp@recommended` of the same backend, tier and context.

| backend | tier | context | path | decode ×off | decode ×MTP (min) | request ×MTP | beats MTP on every prompt (decode / request) |
|---|---|---|---|---|---|---|---|
| candle-cuda | q4 | cold | block_draft:dflash2:linear@recommended | 1.49 | 0.88 (0.76) | 0.87 | False / False |
| candle-cuda | q4 | cold | block_draft:dspine:linear@recommended | 1.48 | 0.87 (0.72) | 0.88 | False / False |
| candle-cuda | q8 | cold | block_draft:dflash2:linear@recommended | 2.00 | 0.98 (0.83) | 0.98 | False / False |
| candle-cuda | q8 | cold | block_draft:dspine:linear@recommended | 1.92 | 0.94 (0.75) | 0.95 | False / False |
| candle-cuda | q4 | reused | block_draft:dflash2:linear@recommended | 1.26 | 0.78 (0.64) | 0.81 | False / False |
| candle-cuda | q4 | reused | block_draft:dspine:linear@recommended | 1.32 | 0.82 (0.76) | 0.84 | False / False |
| candle-cuda | q8 | reused | block_draft:dflash2:linear@recommended | 1.82 | 0.91 (0.80) | 0.94 | False / False |
| candle-cuda | q8 | reused | block_draft:dspine:linear@recommended | 1.70 | 0.85 (0.71) | 0.85 | False / False |
| mlx | q4 | cold | block_draft:dflash2:linear@recommended | 1.64 | 1.33 (1.06) | 1.29 | True / True |
| mlx | q4 | cold | block_draft:dspine:linear@recommended | 1.60 | 1.29 (0.97) | 1.25 | False / False |
| mlx | q8 | cold | block_draft:dflash2:linear@recommended | 1.81 | 1.16 (0.95) | 1.03 | False / False |
| mlx | q8 | cold | block_draft:dspine:linear@recommended | 1.75 | 1.13 (0.88) | 1.10 | False / False |
| mlx | q4 | reused | block_draft:dflash2:linear@recommended | 1.97 | 1.84 (1.17) | 1.81 | True / True |
| mlx | q4 | reused | block_draft:dspine:linear@recommended | 1.15 | 1.07 (0.49) | 1.00 | False / False |
| mlx | q8 | reused | block_draft:dflash2:linear@recommended | 2.06 | 1.20 (1.00) | 1.19 | False / False |
| mlx | q8 | reused | block_draft:dspine:linear@recommended | 1.83 | 1.07 (0.80) | 1.06 | False / False |
