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

The committed test fixtures are synthetic: tiny random-weight checkpoints written
by the upstream `SupraDiT` class, upstream token ids for four prompts, and SHA-256
digests of the real checkpoint's tensors. No upstream weights are committed.
