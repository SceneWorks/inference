# Qwen3.8-27B and Ternary Bonsai 2 27B source inventory

This inventory belongs to epic sc-23935. It separates facts read from frozen upstream artifacts
from compatibility conclusions about this repository. The checked-in Qwen fixtures are exact small
upstream files; they are test inputs, not a redistributed model.

## Frozen sources and licenses

| Artifact | Frozen revision | Relevant files and SHA-256 | License |
| --- | --- | --- | --- |
| `Qwen/Qwen3.8-27B` | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | `config.json` `191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab`; `generation_config.json` `e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e`; `chat_template.jinja` `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`; `tokenizer_config.json` `b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27`; weight index `77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df` | Apache-2.0 |
| `prism-ml/Ternary-Bonsai-2-27B-mlx-2bit` | `3f926b415992eaa2ae9dd7b573706494d6bbf787` | `config.json` `238de7c512cc56a733421e3fd011d88f8260739e3d00e32c5d65b7943cc9f837`; `hadamard.json` `7132a3ec364f0bdac1f08f905f24f0ad2f14245060f592637a0396826d3b5fe6`; `generation_config.json` `cddd0dbd24dbf13229f872b4100c5a0bfe5186eeb09e171b2fc2bac701e98034`; `chat_template.jinja` `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041` | Apache-2.0, with upstream `NOTICE.txt` |
| `prism-ml/Ternary-Bonsai-2-27B-gguf` | `6ed5e12bf84b7a63069882c91dd9e9218647d17b` | published variants `PQ2_0`, `PTQ1_0`, `F16`, plus BF16/Q8_0 multimodal projectors | Apache-2.0, with upstream `NOTICE.txt` |

Pinned source URLs are the Hugging Face repositories named in the table with `/tree/<revision>`.
The Qwen config, generation config, and template are retained in `docs/reference/qwen38/`. The
repository fixtures preserve upstream content and add a final newline where one was absent; their
SHA-256 values are respectively `9a719fad1e0b19b2379bf7cb293fe5984ce52608b80d737fd49bfb91f8887209`,
`e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e`, and
`514d5304fcac63bda60e6faf860d5657010cf43cbc4c633c8bf678ee184e7a37`.

## Qwen3.8 published surface

Confirmed from the frozen artifacts:

- It identifies as `Qwen3_5ForConditionalGeneration` / `qwen3_5` with a nested
  `qwen3_5_text` decoder: 64 layers, 3 linear-attention layers per full-attention layer, hidden
  width 5120, dense intermediate width 17408, 24 query heads, 4 KV heads, head width 256, and a
  262144-token configured context.
- The VLM wrapper has a 27-layer vision tower, 16x16 spatial patches, temporal patch size 2,
  spatial merge 2, image token 248056, and video token 248057. Images, multiple images, and video
  are part of the published input surface.
- The tokenizer config carries the official template and a 262144 model maximum. The template
  exposes thinking on/off, `reasoning_effort` (`xhigh` default, `medium`, `low`),
  `preserve_thinking` (true by default), structured image/video content, tools, assistant tool
  calls, and tool responses. Any other reasoning-effort string raises a template error.
- Generation EOS is the ordered set `[248046, 248044]`; 248044 is also BOS/padding.
- `mtp_num_hidden_layers` is 1. The weight index contains 15 `mtp.*` tensors: one decoder layer,
  attention/MLP norms and projections, `fc`, final norm, and embedding/hidden pre-FC norms.

Confirmed current implementation coverage:

- Both `mlx-llm` and `candle-llm` dispatch `qwen3_5` to the hybrid `Qwen35` decoder and parse the
  dense dimensions above. Both load the same nested language-model and vision namespaces.
- Both providers already resolve generation EOS from `generation_config.json`, before config
  fallback. Both have native Qwen image/video preprocessing, placeholder expansion, vision fusion,
  and multimodal RoPE paths.
- Before sc-23936, the shared request/template contract could pass `enable_thinking` and tools but
  could not express `reasoning_effort` or `preserve_thinking`.
- Neither Qwen loader reads `mtp_num_hidden_layers` or any `mtp.*` tensor. The ordinary decoder can
  load without asking for those keys, so this is an unimplemented speculative-decoding capability,
  not proof that MTP is safe to ignore. sc-23936 owns an explicit native MTP support/refusal policy;
  sc-23942 owns reproducible MTP evidence; sc-23943 owns real-weight terminal acceptance.

Inference based on matching config and key layouts: the existing dense Qwen35 base decoder should
load the non-MTP Qwen3.8 text weights. This remains unproven until a frozen real-weight run. Image
and video quality is also unproven even though the structural path exists.

## Ternary Bonsai 2 published surface and ownership

Confirmed from the frozen MLX config/runtime metadata:

- `model_type` is `prism_hadamard_qwen35`, based on `qwen3_5`. Text and the dense vision tower are
  present; image/video token IDs and processor geometry match the parent. Its config disables MTP
  (`mtp_num_hidden_layers: 0`).
- The MLX artifact declares affine 2-bit groups of 128 and block-1024 normalized Sylvester
  Walsh-Hadamard transforms with explicit signs. Packed coverage includes embeddings, attention,
  linear-attention, MLP projections, and the LM head. The published runtime applies an inverse
  transform after embedding lookup.
- The GGUF repository publishes distinct `PQ2_0` and `PTQ1_0` files plus multimodal projector
  files. These are separate formats and require explicit capability detection.
- The supplied Python vision adapter sets its video processor to `None`. This does not remove video
  from the epic: native video processing and temporal parity must be implemented and tested.
- The frozen artifact is internally inconsistent about vision: `config.json` declares the vision
  component and `vision_artifact.py` wires an FP16 tower, while `PACK-RUNTIME.md` says vision is not
  included. Treat neither statement as acceptance evidence. sc-23939 must resolve the actual tensor
  inventory and execute the native image/video paths; sc-23943 must repeat that with frozen real
  weights. The same runtime note says MTP is absent, consistent with `mtp_num_hidden_layers: 0`.
- The frozen packed inventory contains 402 packed modules, 401 forward transform matrices, and one
  inverse embedding transform. Packaged Gated DeltaNet tensors are already grouped/reordered, so a
  native loader must preserve that layout instead of applying the upstream conversion twice.

Confirmed current gaps and story ownership:

| Capability | Current gap | Owning slice |
| --- | --- | --- |
| Qwen dense text/config/template/stop | Exact Qwen3.8 conformance and real-weight execution remain to be proven | sc-23936; real weights sc-23943 |
| Qwen MTP | No config validation, tensor load, draft execution, or explicit refusal | sc-23936; evidence sc-23942/sc-23943 |
| Bonsai MLX packed text | Unknown model type; no 2-bit projection/embedding/head or Hadamard primitive; current projection only supports dense/Q4/Q8 | sc-23937 |
| Bonsai Candle CPU/CUDA packed text | Same primitive/loader gap; GGUF PQ2_0/PTQ1_0 need explicit routing | sc-23938 |
| Native images and video for both models/backends | Existing Qwen path needs exact Qwen3.8 proof; Bonsai namespace/projector and video parity are absent | sc-23939 |
| ChatWorks import, controls, streaming, tools, lifecycle | Consumer/runtime integration is not in this repository slice | sc-23940 |
| ChatWorks image/video ingestion | End-to-end file/URL and lifecycle flow remains open | sc-23941 |
| Matched capability/quality/performance evidence | No stationary matched runner or results yet | sc-23942 |
| Full real-weight/platform acceptance and immutable pin | No terminal evidence or delivery yet | sc-23943 |

Confidence is high for the artifact inventory and code-path gaps because they come from frozen files
and exact source reads. Confidence is medium for dense Qwen3.8 compatibility until real weights run.
