# Handoff — epic 3040 (mlx-gen advanced video modes + SVD)

Status as of this handoff. Epic: https://app.shortcut.com/trefry/epic/3040

## Where the work lives
- **Branch:** `claude/busy-dhawan-6f3287` → **PR [#156](https://github.com/michaeltrefry/mlx-gen/pull/156)** (open, **MERGEABLE**, **CI green** — fmt+clippy+test, 3m38s).
- Worktree this was built in: `.claude/worktrees/busy-dhawan-6f3287` (a worktree of `mlx-gen`; main is `michaeltrefry/mlx-gen`).
- Reference repos: torch worker `~/Repos/SceneWorks/apps/worker/scene_worker/video_adapters.py`; LTX `ltx_core`/`ltx_pipelines` at `~/Repos/Wan2GP/models/ltx2`; diffusers/transformers source in the venv below.
- **Reference python env:** `~/repos/mflux/.venv-0312` (MLX 0.31.2-matched). This session **installed `einops`, `diffusers` (0.37.1), `transformers` (5.10.1)** into it for golden dumps. SVD + IC-LoRA + Wan/LTX checkpoints are all in the HF cache.

## Read these first
- `docs/SPIKE_ADVANCED_VIDEO_3040.md` — the GO/NO-GO spike + the two conditioning mechanisms.
- `docs/SVD_PORT_SPEC.md` — **exhaustive diffusers spec** (weight keys + building blocks) for the remaining SVD slices. This is the source of truth for S1/S3/S4.

## DONE + validated (in PR #156)
- **sc-3050** spike (Done). **sc-3051** conditioning framework (Done) — `Conditioning::{Keyframe,VideoClip,ControlClip}` + `ReplacementMode` + `GenerationRequest::{keyframes,video_clips,control_clip}` + `Conditioning::kind()`.
- **sc-3052** first_last_frame + extend_clip + video_bridge on **LTX** (In Review). FLF fully usable; extend/bridge = token-native IC-LoRA keyframe-append. The append op is **byte-exact vs torch `ltx_core` `VideoConditionByKeyframeIndex`** (`mlx-gen-ltx/tests/keyframe_cond_parity.rs`).
- **sc-3053** replace_person on **LTX** (In Review). Gray-118 mask op **byte-exact vs Pillow** (`replace_mask_parity.rs`); reuses the append path. Production LTX IC-LoRA confirmed loadable via the existing seam (`adapters::tests::ic_lora_union_control_keys_map_to_av_blocks`).
- **sc-3357** Wan-native first_last_frame on TI2V-5B (In Review) — `build_ti2v_multi_mask` + `build_ti2v_keyframe_z` + a `Conditioning::Keyframe` path; structural tests only (no Wan reference exists for these modes).
- **SVD S0 (sc-3371, Done)** — `mlx-gen-svd` crate + config + EDM scheduler. **Validated vs diffusers `EulerDiscreteScheduler`** (`scheduler_parity.rs`).
- **SVD S2 (sc-3373, Done)** — ViT-H image encoder (reuses sdxl `ClipVisionEncoder` + projection head). **Validated vs transformers** (`image_encoder_parity.rs`, `--ignored`, f32, 0.2% peak-rel).

## REMAINING WORK

### 1. Finish the SVD port (sc-3054 umbrella) — the biggest item
Build against `docs/SVD_PORT_SPEC.md`. Reuse: `mlx-gen-sdxl` `ResnetBlock2D` + 2D VAE encoder pattern; `mlx_gen::nn::conv3d` (per-axis stride/pad) for the temporal `(3,1,1)` convs. Golden pattern: a `tools/dump_svd_*_golden.py` + a `mlx-gen-svd/tests/*_parity.rs`, comparing in **f32** (the dumps run the diffusers component in float32).
- **S1 (sc-3372)** — `AutoencoderKLTemporalDecoder`: 2D encoder (reuse) + **temporal decoder** (net-new: conv_in → `MidBlockTemporalDecoder` → 4× `UpBlockTemporalDecoder` → conv_out → `time_conv_out` Conv3d(3,1,1)). Uses `SpatioTemporalResBlock` (spatial `ResnetBlock2D` + temporal Conv3d block + `AlphaBlender` `time_mixer.mix_factor`). Golden vs diffusers VAE encode + chunked decode.
- **S3 (sc-3374)** — `UNetSpatioTemporalConditionModel` (the big one, ~1428 weights): spatial blocks reuse sdxl unet; net-new = `SpatioTemporalResBlock`, `TransformerSpatioTemporalModel` (spatial `BasicTransformerBlock` + `TemporalBasicTransformerBlock` + AlphaBlender), micro-cond (`added_time_ids` [fps−1, motion_bucket_id, noise_aug] → add_embedding 768). Per-block + full-forward golden.
- **S4 (sc-3375)** — pipeline (image→CLIP embed + noise-aug VAE-encode→4-ch concat→8-ch UNet input; frame-wise CFG `linspace(min,max,frames)`; chunked temporal decode) + register the `svd_xt` provider (Modality::Video, image_to_video via `Conditioning::Reference`; motion_bucket_id/fps/noise_aug via request fields) + e2e parity vs `StableVideoDiffusionPipeline`.

### 2. Wan-VACE port (sc-3388) — the IC-LoRA-type pose/depth control on Wan
Full model port (VACE context/hint blocks on the Wan DiT). Covers pose/depth/sketch control + the **Wan** side of extend/bridge/replace_person. Torch ref = diffusers `WanVACEPipeline` (used by `video_adapters.py` for Wan replace_person). Pick the checkpoint (Wan2.1/2.2-VACE), consume `Conditioning::ControlClip` + reference images + `conditioning_scale`. Slice like SVD.

### 3. Gated / cross-repo (not mlx-gen code, or external deps)
- **extend/bridge/replace_person e2e parity** — the conditioning ops are byte-validated and the IC-LoRA loads, but a full-render byte-parity vs torch `ICLoraPipeline` is impractical (22B torch base; mlx-gen runs the AV-q4 base, not the Lightricks 22B distilled). A **directional** e2e render (load AV-q4 via `LTX_BASE_DIR` + the Union-Control IC-LoRA via `spec.adapters` + a `VideoClip` request) is possible if quality confirmation is wanted.
- **sc-3385** (chore) — investigate + fix the SceneWorks worker's "all advanced modes route to LTX" (`_uses_ic_lora_pipeline`); define the per-mode×per-model routing matrix. **SceneWorks worker repo**, own session.
- **sc-3055** (chore) — routing + cutover: route modes/SVD to the Rust worker, retire the torch video adapters. **SceneWorks worker repo**, own session; blocked by SVD + the e2e parity pass.

## Gotchas / lessons
- **CI won't run while the PR conflicts with main** — GitHub can't build the `pull_request` merge ref, so it silently skips. If runs stop appearing, check `gh pr view <n> --json mergeable` and merge main.
- CI = `cargo fmt --all --check` + `cargo clippy --workspace --all-targets -- -D warnings` + `cargo test --workspace` (macOS/Metal, `RUST_TEST_THREADS=1`). Run all three locally before pushing; clippy `-D warnings` is strict (needless borrows of temporaries, doc lines starting with `+ `, needless lifetimes).
- LTX DiT forward is **fully token+positions driven** (RoPE from `positions`), so appended conditioning tokens need no grid — that's what makes the IC-LoRA append path work.
- Golden dumps live in `tools/dump_*_golden.py` (use `from _paths import fixture`); parity tests in each crate's `tests/`. `Weights::cast_all(Dtype::Float32)` to gate in f32.
- Memory file: `~/.claude/projects/-Users-michael-Repos-mlx-gen/memory/advanced-video-epic3040.md`.
