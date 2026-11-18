# SenseNova-Loop-9B on Candle

The provider reads the 11 indexed target shards of Hugging Face revision
`3e1ba6e8ac98090f9f7c03a55f26269eb3850fb6`. Preparation verifies the
original shard SHA-256 values. Loading checks every tensor name, dtype and
shape against the 857-tensor release schema and verifies the tokenizer,
template, generation settings and image processor sidecars. The upstream
conditional-generation wrapper ignores the separate MTP companion and its
inert training gate. Preparation retains the gate in its original precision.

## Resident weights and load lifetime

Candle prepares one shard at a time on the CPU. Each of the 201 selected
text projections, including the untied LM head, replaces its original BF16
matrix under the same key. Q8 uses GGML Q8_0 blocks of 34 bytes per 32
values. Q4 uses Q4_K blocks of 144 bytes per 256 values. All selected input
widths are divisible by 256. Both tiers retain BF16 embeddings and vision
weights and the original FP32 mHC weights, biases and DeltaNet decay
parameters. The loader reconstructs each packed projection on the requested
device from its host blocks. Source blocks stay in host memory during
construction and the device holds one packed copy. The decoder holds one
set of weights for each of its 32 physical layers.

| Tier | Tensor payload bytes from the release geometry |
| --- | ---: |
| BF16 | 18,832,231,392 |
| Q8_0 | 11,398,482,912 |
| Q4_K | 7,433,817,056 |

These counts exclude file headers, device allocator reservations, temporary
casts and request state. The runtime widens the zero-centered normalization
multipliers before adding one, retains router weights in FP32 and executes
the vision tower in the GPU compute dtype.

## Request lifetime

The 56 visits allocate 14 independent full attention K/V pairs and 42
independent DeltaNet conv/SSM states. At batch one on CUDA, the former cost
57,344 bytes per position. Each linear slot holds a BF16 convolution tail of
shape `[1,3,8192]` and an FP32 SSM of shape `[1,32,128,128]`, totaling
90,144,768 bytes across all 42 slots for one retained state. Rewindable caches
charge the checkpoint ring depth. The model shares physical weights across
the two passes and retains separate state for each visit.

Within a routed block, attention and FFN each normalize two residual streams
in FP32. The branch consumes a single hidden vector after applying pre-gates
in activation precision. The mix matrix occupies four FP32 values per token.
Each branch releases its row/column-normalization temporaries after 20
Sinkhorn iterations and casts the mix and post-gates for the residual update.
Request admission charges two residual widths in its activation and graph
workspace estimates. Vision admission also charges decoded pixels, patch
rows, encoder workspace and decoder splice inputs. These estimates do not
constitute a measured device peak.

## Production-shape proof

The ignored CUDA test `sensenova_loop_release_proof` accepts the three
`SENSENOVA_LOOP_{BF16,Q8,Q4}_SNAPSHOT` directories and a frozen upstream
oracle JSON at `SENSENOVA_LOOP_ORACLE`. The test loads one tier at a time.
It renders a text question and red and blue 256×256 image requests with the
production tokenizer, processor, vision tower and 4096-wide decoder. It compares
expanded prompt IDs, release EOS, tokenizer thinking modes, selected raw
logits after prefill and two teacher-forced continuations, cold versus cached
continuation logits, distinct live cache addresses, generated answers, strict
JSON and streaming cancellation. The BF16 parity bound comes from the upstream
oracle. The proof records the delta for each quantized tier and uses the same
tier's cold prefill for its cache comparison. Device proof receipts and
measured memory belong to the dedicated workflow.

The BF16 and Q4 CUDA legs also make a 1536×1024 RGB request as the first
request after a fresh provider load. Both use the same red image, prompt,
greedy settings and one-token budget. Their `SENSENOVA_LOOP_MEMORY_ENDPOINT`
receipts contain the expanded prompt count, sampled ID, generated count,
synchronized device-wide usage, and CUDA primary-context pool used/reserved
high watermarks reset separately before loading and before the request.
