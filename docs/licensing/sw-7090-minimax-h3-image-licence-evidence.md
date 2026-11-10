# MiniMax-H3-Image licence evidence (SW-7090)

Read on 2026-11-10. The repository is not gated.

| Item | Primary evidence | SHA-256 of the file read |
| --- | --- | --- |
| Component `minimax_h3_image` | [README at 492fd8c4](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/492fd8c4eca666f1085f57e30738c1e0d86b3439/README.md): front matter `license: other`, `license_name: minimax-h3-image-community-license-agreement`, `license_link: LICENSE` | `eba19a0c6f71a5b1871277fa0bc6689c0a58086956b34fb3f940765edde66469` |
| Family `minimax-h3-image-community` | The LICENSE at the same revision, as read and transcribed by SW-7099 (`sw-7099-minimax-h3-image-pe-licence-evidence.md`) | `edfab8ece9d0ccd39106c4fe44105ce599b27ca7dd22601b736a8e9219cb4d26` |

The row covers what `mlx-gen-minimax-h3`'s `minimax_h3_image` generator loads from this
repository: `transformer_diffusers/` (the DiT), `vae_diffusers/` (the 2-D VAE) and `tokenizer/`.

The Qwen3-VL-32B text encoder is rowed separately as `qwen3_vl_32b_instruct` (Apache-2.0), as for
MiniMax-H3: the generator reads it from MiniMax-H3's `text_encoder/` (or this repository's own
copy), both byte-identical to upstream `Qwen/Qwen3-VL-32B-Instruct`.
