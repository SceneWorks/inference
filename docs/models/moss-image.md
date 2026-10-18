# MOSS-Image native contract (sc-26322)

This foundation prepares native MLX and Candle providers; it does not register an executor.
`gen_core::moss_image` is the backend-neutral authority for model identity, request resolution,
precision policy, advanced inputs, tensor coverage and author sampling. Python is not a
production dependency. Backend providers must call `MossImageResolved::resolve` before
allocation and implement `MossImageAdvancedGenerator` beside the ordinary `Generator` API.

## Frozen authority and assets

| Model | Repository | Revision | License authority |
| --- | --- | --- | --- |
| Base | OpenMOSS-Team/MOSS-Image-Base | 1ceede80badddd6a7812069619edaa5633587192 | bundled LICENSE, Apache-2.0 |
| SFT | OpenMOSS-Team/MOSS-Image-SFT | fc5de0d56fc910cbca6f6625a70f5f582b26deb1 | bundled LICENSE, Apache-2.0 |
| Optional 8-step distillation | lodestones/MOSS-Image-SFT-8-steps-dmd | 5ff314b4c1470876b114b810146085db8a181a6e | README metadata, Apache-2.0; package has no separate LICENSE file |

Primary sources: [frozen pipeline](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT/blob/fc5de0d56fc910cbca6f6625a70f5f582b26deb1/pipeline_moss_image.py),
[frozen tokenizer template](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT/blob/fc5de0d56fc910cbca6f6625a70f5f582b26deb1/tokenizer/chat_template.jinja),
[frozen distillation instructions](https://huggingface.co/lodestones/MOSS-Image-SFT-8-steps-dmd/blob/5ff314b4c1470876b114b810146085db8a181a6e/README.md).
Pipeline SHA256: `1dac1aca9feb1bfdfc727a401e4b67d61a505d3190706d1b7d2d784f88678b31`.
Small frozen config bytes and their source hashes live in
`crates/contracts/gen-core/tests/fixtures/moss_image/`. The golden fixture records its provenance.
They contain no weights or generated images. Its 50-step schedule is retained from an existing
successful author-pipeline receipt; that historical run supplies scalar expected values, not
native acceptance or a quality claim.

`SNAPSHOT_FILES` enumerates all required Base/SFT files: three transformer shards/index/config,
four encoder shards/index/config, tokenizer/config/chat template, scheduler config, VAE
weights/config and LICENSE. This is a complete role inventory, not a requirement to duplicate
shared weights in the Base directory: a Base transformer-only snapshot uses the exact SFT
shared snapshot passed as `LoadSpec.components["moss-shared-snapshot"]`. SFT may likewise pass
that component explicitly. Never discover it from a user cache or environment variable.
Provision by exact revision and bind actual file contents through
existing prepared source pins and shard-index validation. Repository/revision strings alone
are not file-content verification. Never execute the package's Python code at runtime.

The Qwen3.5 encoder has 32 hybrid blocks, hidden width4096, and a `language_model` decoder
root. Its checkpoint has no language-model head: use `from_weights_headless` and
`forward_hidden`, with fresh per-prompt state and no padding during the encoder forward.
`ChatTemplate::MossImageQwen35` matches the single-user template: trim outer user whitespace,
preserve the content inside it, add the assistant generation prefix and `<think>\n`.
Right truncation includes template tokens and is limited to1–4096 tokens. No generated
reasoning is sampled. Encoder hidden rows are final-normalized, not vocabulary logits.

The VAE is AutoencoderKLFlux2 with 32 raw /128 packed channels, FP32 BN statistics and decode,
epsilon1e-4. Encoder widths are `[128,256,512,512]`; decoder widths are
`MOSS_VAE_DECODER_CHANNELS=[96,192,384,384]`. Pass the latter explicitly to the reusable
config-driven FLUX2 decoder. Packed channels split `(c,ph,pw)` with `c*4+ph*2+pw` order.
Do not load an ordinary large-decoder VAE or substitute a Qwen3 encoder.

## Requests and complete native surface

Provider IDs are `moss_image_base` and `moss_image_sft`. A fresh
`GenerationRequest::default()` has `moss_image=None`; `MossImageOptions::default()` has
acceleration `Off`. Supplying optional components does not turn acceleration on.
Ordinary defaults are50 requested steps, CFG3, shift5, training shift1, negative prompt empty,
unconditioned skipped block22, SLG off, context4096. An explicit empty skip list disables
unconditioned skipping. Indices0–41, nonnegative finite scales and
`0 <= start <= stop <= 1` are checked. The interval is measured in actual deduplicated
sampling steps; it controls both unconditioned skipping and the extra positive SLG branch.
CFG is used whenever scale differs from1, including0. CFG=1 plus SLG uses the author's
`skipped + scale*(conditional-skipped)` formula.

Preserve text, multilingual content, tag sequences and compact JSON/layout strings as prompt
content. Layout is text conditioning (`summary.caption`, `items`, `bbox_format:"yxyx"`,
`bbox_coord_space:"abs_centralized"`), with generation dimensions matching its canvas.
No reference-image editing or control tensors are exposed by this frozen model.

The generic request carries prompt/prompt_batch, count, dimensions (positive multiples16),
seed, steps, guidance/negative prompt and optional typed MOSS controls. The native advanced
trait additionally accepts positive/negative FP32 embedding buffers `[B,S,4096]`, optional
binary masks `[B,S]`, per-prompt negative text batches, supplied FP32 NCHW noise `[B*count,128,H/16,W/16]` and latent output.
Each mask row must retain a token. Embeddings and text are mutually exclusive; an empty
prompt is a valid actual text prompt when no embeddings are supplied. Upload embeddings in
the intended encoder dtype and compact masked tokens per sample, preserving order.
`validate` returns total output count after prompt-batch × image-count expansion. Native
providers preserve per-image seed/provenance and cancellation/progress through normal runtime
contracts. The backend owns device buffers; these advanced inputs provide a portable CPU
interchange without introducing tensor types into gen-core.

## Author sampling and precision

`moss_image_schedule` implements V2 nearest-training-grid Euler: construct the shifted
inference grid, snap to the1000-point training grid, choose the lower sigma on exact ties,
remove consecutive duplicates, append zero. Feed `sigma*1000` to the transformer. Use
requested steps for schedule construction and actual step count for progress/SLG. The fixed
noise, CFG/SLG and Euler updates are FP32, as is VAE decoding. Official DiT and text encoding
are BF16. RoPE frequencies/phase/sin/cos are FP32 before casting to activation dtype;
`moss_time_features` computes the time sinusoid in FP64 then casts once to FP32.

Acceleration is SFT-only. With `Dmd(UnmergedBf16)`, require exact SFT parent plus
`moss-dmd-lora`, `moss-dmd-time`, `moss-dmd-dataspace` named `LoadSpec.components`.
The matching files are `lora.safetensors`, `time_embedding.safetensors`,
`dataspace.safetensors`. `validate_dmd_package` requires exact tensor coverage, shapes,
finite FP32 values, rank128/alpha128 on all six projections in all42 blocks, all four
time-MLP tensors and both shared/head modulation biases. The low-rank output remains FP32
and is added to the backbone output after promoting that output to FP32. Keep the time MLP
and both full modulation modules FP32. Do not BF16-merge, narrow the residual before adding,
or apply only the time tensors without their companion biases.

`Dmd(FullFp32)` selects the author-published full FP32 distilled transformer (four
shards/index/config), plus dataspace. It must load AND execute FP32, without autocast.
The named components are `moss-dmd-transformer` and `moss-dmd-dataspace`. This is a precision
option, not another model variant. The package also includes `generator.safetensors`, a
redundant combined representation; the native canonical unmerged route uses the separate
files for exact key validation. Neither published precision route asserts equality to the
other: the historical mixed-vs-full precision threshold result is not an adoption gate.

For either DMD precision, validate FP32 finite `latent_mix.Q [128,128]` and `step_codes [8]`.
Accept trained prefixes1–8, default8. There is no CFG, Euler, negative prompt, scheduler,
SLG, layer skipping or selectable shift in this mode. Start `x0=0`, retain the same noise z,
then for each selected code compute `x=z+Q*x0`, predict v, and replace `x0=z-v`.
`moss_dmd_input` / `moss_dmd_update` specify that recurrence and matrix orientation;
backends implement equivalent device operations. Never blend fresh noise or Euler sigma
deltas into it. Decode final x0 through the same VAE.

Every cache/load identity includes the selected variant, acceleration/precision policy and
all exact content pins. OFF→ON→OFF requires separate immutable loaded state or a full reload;
no ordinary request may inherit DMD LoRA/time replacements from cached accelerated state.
Reject absent components and incompatible parent/mode/control combinations before execution.
