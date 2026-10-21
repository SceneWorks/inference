# Supra2-IMG component licence evidence (sc-24213)

Read on 2026-10-20 from the model cards at the exact revisions the provider pins
(`gen_core::supra2::{SUPRA2_SOURCE, FLAN_T5_BASE, SD_VAE}`). None of the
repositories is gated and none ships a separate LICENSE file; each card's front
matter is the declaration.

| Component | Primary evidence | Declaration | Card SHA-256 |
| --- | --- | --- | --- |
| `supra2_img` | [README at 10dec6e4](https://huggingface.co/SupraLabs/Supra2-IMG/blob/10dec6e4b4b5d1c44fd1d7d3fe5e50137333da5b/README.md) | Front matter `license: apache-2.0` | `669f85ac6f8a9f00e672aaae2164c986ec35a76fda8f8a3fb592dd0ed78cc6b0` |
| `flan_t5_base` | [README at 7bcac572](https://huggingface.co/google/flan-t5-base/blob/7bcac572ce56db69c1ea7c8af255c5d7c9672fc2/README.md) | Front matter `license: apache-2.0`; body "License: Apache 2.0" | `b6fb54f6831f3677f537cdb83d6f822e8705d203c605a9ec4c8192bda5a89828` |
| `sd_vae_ft_mse` | [README at 31f26fde](https://huggingface.co/stabilityai/sd-vae-ft-mse/blob/31f26fdeee1355a5c34592e401dd41e45d25a493/README.md) | Front matter `license: mit` | `8ac69483c18a45d0fee531e08b6ede97ee9b4c79e875659c1b0e6a5903ca3400` |

The rows reuse the existing `apache-2-0` and `mit` families. The `supra2_img`
provider row lists all three components because a load reads all three: the
converted denoiser bundle (derived from `model_final_ema.pt`), the Flan-T5-Base
encoder/tokenizer and the SD VAE. The converted bundle is produced locally at
install time from the pinned checkpoint and is not redistributed. Metadata is
disclosure and does not gate execution.

Most committed test fixtures are synthetic: `tiny_model_with_uncond.pt`,
`tiny_ema_without_uncond.pt`, `tiny_digests.json`,
`mlx-gen-supra2/tests/fixtures/tiny_dit_parity.safetensors`, and
`mlx-gen-supra2/tests/fixtures/tiny_t5_parity.safetensors` contain deterministic
random-weight models, inputs, and torch outputs. The two tiny `.pt` files are
written by the frozen upstream `SupraDiT` class, but contain no upstream parameter
or output. `gen-core/tests/fixtures/supra2/supra2_reference_digests.json` contains
only SHA-256 digests of real Supra2 checkpoint tensors. `tokenizer_cases.json`
contains four prompt strings and their ids/masks emitted by the pinned Flan-T5-Base
tokenizer; it contains neither tokenizer files nor tensors.

One committed fixture is deliberately different:
`crates/media/candle-gen/candle-gen-supra2/tests/fixtures/supra2_real_golden.safetensors`.
It contains the `t5.uncond_text` `[128, 768]` F32 and `t5.uncond_mask` `[128]` F32
tensors copied verbatim from the Supra2-IMG `model_final_ema.pt` checkpoint. It
also contains torch CPU F32 reference inputs and outputs: Flan-T5 contexts/masks
for the four tokenizer cases (`t5.ctx.*` / `t5.mask.*`), a Supra2 DiT forward
(`dit.*`), VAE decode inputs/outputs (`vae.*`), and four full sampling trajectories
with decoded pixels (`traj.*`). The `t5.ctx.*`, `dit.*`, `vae.*`, and `traj.*`
values are derived from real Supra2-IMG, Flan-T5-Base, and SD-VAE weights; the
trajectory pixel tensors are stored as U8.

Accordingly, redistribution of this golden is disclosed with attribution to
SupraLabs/Supra2-IMG (Apache-2.0), google/flan-t5-base (Apache-2.0), and
stabilityai/sd-vae-ft-mse (MIT), at the revisions and evidence links in the table
above. The direct checkpoint tensors are Supra2-IMG material; the real-weight
reference outputs retain the applicable attribution and notice obligations of the
components used to produce them. No other upstream weight tensor is committed.
