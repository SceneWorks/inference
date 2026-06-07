# Wan-VACE port spec (epic 3040 / sc-3388)

Native MLX port of **Wan-VACE** (Video All-in-one Creation and Editing) onto the existing
`mlx-gen-wan` crate. VACE is the Wan equivalent of LTX IC-LoRA control: `control video + mask +
reference images → controllable generation`, covering **pose / depth / sketch control, extend,
video_bridge, and replace_person** through one model. The SceneWorks torch worker already routes the
Wan `replace_person` path to diffusers `WanVACEPipeline` (`video_adapters.py`), so this port covers
that plus the IC-LoRA-type pose/depth control.

**Reference:** diffusers 0.37.1 — `WanVACETransformer3DModel`
(`models/transformers/transformer_wan_vace.py`) + `WanVACEPipeline`
(`pipelines/wan/pipeline_wan_vace.py`), both present in `~/repos/mflux/.venv-0312`.

**Key finding — VACE is purely additive on the base Wan DiT.** The base `WanTransformer3DModel` is
unchanged; VACE adds (a) one `vace_patch_embedding`, (b) a small list of `WanVACETransformerBlock`s
that produce per-layer "hints", and (c) hint-injection into the main block residual stream. The
`mlx-gen-wan` crate already implements the entire base DiT (`Block`, patchify→`patch_embedding`
linear, `rope`, `condition_embedder`, `norm_out`/`proj_out`, `scale_shift_table`), the z16 Wan VAE,
and UMT5 — so this port reuses all of it and adds only the VACE pieces.

---

## 1. Checkpoint (first task — confirm against the worker)

Diffusers' default `WanVACETransformer3DModel` config is the **14B** (num_layers=40,
num_attention_heads=40, attention_head_dim=128 → inner_dim=5120, ffn_dim=13824,
vace_layers=[0,5,10,15,20,25,30,35], vace_in_channels=96). The lighter **Wan2.1-VACE-1.3B** is the
pragmatic target for local validation (smaller download). Both are config-driven — the port must read
dims from `config.json`, not hardcode.

- **Worker finding (`~/repos/SceneWorks/apps/worker/scene_worker/video_adapters.py`):** the worker
  uses VACE **only for `request.mode == "replace_person"`** on the `wan_video` adapter
  (`_pipeline_kind → "vace"`, line ~1981), via diffusers `WanVACEPipeline` (line ~1885), passing
  `reference_images` + `conditioning_scale` (lines ~2087-2090). The repo is request-driven
  (`advanced.modelRepo` override, else the Wan target repo). **It loads via
  `WanVACEPipeline.from_pretrained` → the checkpoint is diffusers-layout** (e.g.
  `Wan-AI/Wan2.1-VACE-1.3B-diffusers` / `-14B-diffusers`), NOT the native Wan `.pth` layout the existing
  `mlx-gen-wan` converter (`convert.rs`/`pth.rs`) targets.
- **Naming/loader decision (S0):** the base `mlx-gen-wan` DiT uses native Wan naming
  (`blocks.{i}.self_attn.q/k/v/o`, `modulation`, `ffn.fc1/fc2`, `patch_embedding_proj`, `head.head`).
  The VACE checkpoint is diffusers-named (`blocks.{i}.attn1/attn2`, `scale_shift_table`,
  `ffn.net.0.proj`, `vace_blocks.{i}.proj_in/proj_out`, `patch_embedding`, `vace_patch_embedding`).
  So the port needs a diffusers-VACE → native-layout mapping (extend `convert.rs`) **or** a
  diffusers-layout loader. Recommend: extend the existing Wan converter so the base blocks reuse the
  validated native path and only the vace pieces are new keys.
- **No VACE checkpoint is in the HF cache yet** → real-weight e2e parity is a **provisioning
  dependency** (like the LTX IC-LoRA weights were for sc-3052). Structural parity (below, from a
  randomly-initialized small-config diffusers model) does NOT need it.

## 2. The VACE transformer (the core new engine piece)

`WanVACETransformer3DModel` config adds two fields over the base: `vace_layers`
(default `[0,5,10,15,20,25,30,35]`, must include 0) and `vace_in_channels` (default 96). New modules:

- `vace_patch_embedding`: `Conv3d(vace_in_channels=96, inner_dim, kernel=patch_size, stride=patch_size)`
  — same patchify+project as the base `patch_embedding`, but 96 input channels. In `mlx-gen-wan` this is
  a patchify (unfold the 96-ch control latent into `prod(patch_size)`-sized patches) → linear of
  `in_dim = 96·prod(patch_size)`, mirroring the existing `patch_embedding` linear.
- `vace_blocks`: `len(vace_layers)` × `WanVACETransformerBlock`. Each is structurally a Wan block
  (FP32-LayerNorm `norm1`/`norm2`/`norm3`, `attn1` self-attn with rope, `attn2` cross-attn to text,
  `ffn` gelu-approximate, own `scale_shift_table` [1,6,dim]) **plus**:
  - `proj_in` (Linear dim→dim) **only on block 0** (`apply_input_projection = i==0`): block 0 does
    `control = proj_in(control) + hidden_states` (injects the main noisy-latent tokens into the control
    stream once, at the start).
  - `proj_out` (Linear dim→dim) on **every** vace block (`apply_output_projection=True`): emits the hint
    `conditioning_states = proj_out(control)`.
  - Each block also returns the updated `control_hidden_states` (threaded through all vace blocks
    sequentially).

### Forward (`WanVACETransformer3DModel.forward`)

Inputs: `hidden_states` [B,16,T,H,W] noisy latent, `timestep`, `encoder_hidden_states` (text),
`control_hidden_states` [B,96,T_c,H,W], `control_hidden_states_scale` (per-vace-layer scalar, default
ones), optional `encoder_hidden_states_image` (I2V).

1. `rotary_emb = rope(hidden_states)`.
2. Patch-embed `hidden_states` → tokens [B, L, dim]. Patch-embed `control_hidden_states` via
   `vace_patch_embedding` → [B, L_c, dim], then **zero-pad along tokens** to L (`L − L_c` zero tokens
   appended).
3. `condition_embedder(timestep, text, image)` → `temb`, `timestep_proj` (unflatten to [B,6,dim]),
   `encoder_hidden_states` (+ prepend image tokens if I2V). Shared with the base DiT.
4. **Hint prep:** run the vace_blocks sequentially over `control_hidden_states` (block 0 first does
   `proj_in(control)+hidden_states`); collect `(conditioning_states_j, scale_j)` per vace block; then
   **reverse** the list.
5. **Main blocks:** run the base Wan blocks. At each main layer `i` in `vace_layers`, pop the
   (reversed) list and `hidden_states += control_hint · scale`. (vace_block[j]'s hint is added at main
   layer `vace_layers[j]`.)
6. Output norm (`scale_shift_table` [1,2,dim] + temb) → `proj_out` → unpatchify (same as base Wan).

**Numerics to match (diffusers):** `FP32LayerNorm` (norms computed in f32 then cast back —
`_keep_in_fp32_modules` includes `norm1/2/3`, `time_embedder`, `scale_shift_table`); `qk_norm =
"rms_norm_across_heads"`; `eps=1e-6`; gelu-approximate FFN (the `mlx_gen::nn::gelu_tanh`, NOT
`gelu_approximate` — see the mlx-rs f64-const note). The base block math already matches the Wan T2V
port (byte-validated), so the vace block reuses that machinery + adds proj_in/out + its own
scale_shift modulation.

## 3. Conditioning construction (the pipeline / host side)

The 96-ch `control_hidden_states` = `cat([video_latents(32), mask_latents(64)], dim=channels)`:

- **`prepare_video_latents` → 32 ch.** With a mask: `inactive = video·(1−mask)`, `reactive =
  video·mask`; VAE-encode each (z16, `sample_mode="argmax"` = the mode/mean, not sampled), normalize
  `(x − latents_mean)·latents_std` (per-z-dim, the existing Wan VAE constants), then
  `cat([inactive(16), reactive(16)], dim=channels)` = 32. (No mask → just the encoded video, 16ch,
  then mask defaults to all-ones so inactive=0/reactive=video.)
- **`prepare_masks` → 64 ch.** Mask in [0,1]; `view(F, new_h, 8, new_w, 8).permute(2,4,0,1,3).flatten(0,1)`
  → 64 = `vae_scale_factor_spatial²` channels, then `interpolate(mode="nearest-exact")` to the latent
  temporal length `new_F`.
- **Reference images:** each VAE-encoded to a latent frame, `cat([ref_latent, zeros_like], dim=ch)`
  (32ch) **prepended along frames** to the video latent; the mask gets `num_ref` zero frames prepended.
  So the control sequence is `num_ref` frames longer; the noisy latents include matching ref slots.
- `conditioning_scale` (per-vace-layer, default 1.0) → `control_hidden_states_scale`.

This is host + Wan-VAE work (reuses the existing z16 VAE encode + latents_mean/std). Maps the engine
`Conditioning::ControlClip` (mask + control frames) + reference images + `conditioning_scale` from the
epic-3040 framework.

## 4. Mode mapping (what VACE serves)

- **replace_person** — masked control clip (the worker's masked region) → reactive/inactive split; the
  Wan answer to LTX replace_person (sc-3053). Covers the worker's existing `WanVACEPipeline` path.
- **pose / depth / sketch control** — the control video is the pose/depth/sketch render; mask = all
  active. The IC-LoRA-type control Michael asked for.
- **extend / video_bridge** — control = the source clip frames at the kept positions, mask marks the
  generated span. The Wan answer to the sc-3357 design item.

## 5. Slicing (mirrors the SVD port sc-3371–3375)

- **S0** — scaffold: `WanVaceConfig` (vace_layers, vace_in_channels) read from config.json; module
  stubs; the small-config golden harness (`tools/dump_wanvace_*_golden.py` dumping from a **randomly
  initialized small-config `WanVACETransformer3DModel`** — no big checkpoint needed) + `Float32`
  parity scaffolding.
- **S1** — the VACE transformer: `vace_patch_embedding` + `WanVaceBlock` (proj_in/out + own
  scale_shift) + `forward_vace` (hint prep + injection). **Structural parity** vs the small-config
  random diffusers model (isolated vace-block gate + full-forward gate), f32.
- **S2** — conditioning construction: `prepare_video_latents` (inactive/reactive VAE-encode +
  latents_mean/std) + `prepare_masks` (8×8 unfold → 64ch + nearest-exact temporal interp) +
  reference-latent prepend → 96-ch control. Byte-validate the host ops vs torch.
- **S3** — pipeline + `wan_vace` provider: `denoise` with `control_hidden_states_scale`, wire
  `Conditioning::ControlClip` + reference + `conditioning_scale`, mode routing; **real-weight e2e
  parity (checkpoint-gated)**.

## 6. Validation pattern (reuse from SVD sc-3054)

`tools/dump_wanvace_*_golden.py` at small dims (e.g. num_layers=4, vace_layers=[0,2], dim small) +
`mlx-gen-wan/tests/wanvace_*_parity.rs --ignored`, `Weights::cast_all(Float32)`; isolated-component
gates as tight structural guards + a full-forward gate; bisect a high full-forward gap by dumping
isolated submodule I/O (call the diffusers submodule directly). Real-weight e2e is gated on the VACE
checkpoint (provisioning dependency).
