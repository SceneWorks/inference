# Upstream licence and notice texts (YuE2, sc-22989)

Verbatim copies of every licence / notice text the YuE2 closure and its source carry, so the
licence policy (`src/license.rs`) cites the exact bytes it was read from. Each file is pinned by
SHA-256 in `LICENSE_TEXTS`, and the `yue2-weights/`, `mert2-weights/`, `sheetsage2/` and
`mert-v2-fullsong/` files are byte-identical to the `LICENSE` / `THIRD_PARTY_NOTICES.md` /
`licenses/*.txt` files the inventory pins inside each model snapshot. No model weights are stored
here.

| Directory | Source | Terms |
| --- | --- | --- |
| `yue2-source/` | `github.com/multimodal-art-projection/YuE` @ `92a73cc7652fcc1f937855e4b765e0a0edd7ff2e` | `LICENSE`: Apache-2.0 (first-party source). `MODEL_LICENSE`: CC BY-NC 4.0 for the YuE2-3B / YuE2-Vae / YuE2-Vae-legacy weights only. `THIRD_PARTY_NOTICES.md` + `licenses/`: MIT for the Oobleck VAE (stable-audio-tools) and SnakeBeta (BigVGAN) code. |
| `yue2-weights/` | `m-a-p/YuE2-3B`, `m-a-p/YuE2-Vae`, `m-a-p/YuE2-Vae-legacy` at their pinned revisions (identical in all three) | `LICENSE`: the CC BY-NC 4.0 weight licence. Notices as above (whitespace-only differences from the source copies). |
| `mert2-weights/` | `m-a-p/MERT-v2-FullSong` and `m-a-p/SheetSage2` at their pinned revisions (identical) | CC BY-NC 4.0. Its preamble names only the MERT2 checkpoints; SheetSage2 ships the same file. |
| `sheetsage2/`, `mert-v2-fullsong/` | each repository's `THIRD_PARTY_NOTICES.md` | Dependency and rendering-asset notices. |
| `qwen-tiktoken/` | `Qwen/Qwen-7B` @ `ef3c5c9c57b252f3149c1408daf4d649ec8b6c85` | The Tongyi Qianwen License Agreement and its NOTICE. YuE2's `qwen.tiktoken` is byte-identical to Qwen-7B's, and the YuE2-3B repository ships neither file. |
