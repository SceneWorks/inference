**Acceptance on the identical prefix** — `old-loop-87478b336` (the pre-engine `qwen_mtp` loop, growing KV cache) vs `head-51621c288` (the unified engine, static KV cache), 124 new tokens per row, same fixture / prompt / snapshot (`E:\huggingface\hub\models--Qwen--Qwen3.8-27B\snapshots\1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`), GPU 0.

| K | tokens identical (old vs new) | proposed old / new | accepted old / new | target_forwards old / new | acceptance old / new | syncs/verify new | replay_forwards new | match |
|---|---|---|---|---|---|---|---|---|
| ref | yes (124) | n/a | n/a | 124 / 124 | n/a | n/a | n/a | yes |
| 1 | yes (124 / 124) | 65 / 65 | 57 / 57 | 75 / 75 | 0.877 / 0.877 | 1.0 | 8 | **yes** |
| 2 | yes (124 / 124) | 91 / 91 | 77 / 77 | 57 / 57 | 0.846 / 0.846 | 1.0 | 10 | **yes** |
| 3 | yes (124 / 124) | 117 / 117 | 83 / 83 | 58 / 58 | 0.709 / 0.709 | 1.0 | 17 | **yes** |
| 4 | yes (124 / 124) | 132 / 132 | 90 / 90 | 53 / 53 | 0.682 / 0.682 | 1.0 | 19 | **yes** |
| 5 | yes (124 / 124) | 154 / 154 | 92 / 92 | 54 / 54 | 0.597 / 0.597 | 1.0 | 22 | **yes** |

old binary built from `87478b336daf1fc87f91bda88f78b5ccc3c5638f` (clean tree: True); new from `51621c2885f8158164a61c43f98ba8bcd9504e45` (clean tree: True); co-tenants on GPU 0 at start: old [], new []. `match` = proposed, accepted and target_forwards identical; `tokens identical` = the two rows' generated ids are equal position by position.

Result: every K matches exactly (proposed / accepted / target_forwards) on the identical prefix.
