# Mage-Flow — the six epic-14034 GAPS, resolved from the vendored reference (sc-14036)

Every answer below was read out of the **vendored, frozen** source in `mage_flow/` (upstream
`microsoft/Mage @ df7f84d9f8fc991d189d929f03cff623b430a4a2` — see `VENDORED.md`). Citations are
`file:line` **into this vendored copy**, so they stay valid for as long as the copy does. Where
an answer corrects the epic description, that is called out — the description was written from
the model cards and paper before the code was read.

Line numbers move on a re-vendor. Re-verify them (and these answers) whenever `VENDORED.md`'s
pinned SHA changes.

---

## GAP 1 — TE conditioning layer index

**Answer: the FINAL (36th) decoder hidden state, AFTER the model's final RMSNorm — then drop the
first `drop_idx` system-prompt tokens (34 for gen, 64 for edit).**

**This corrects the epic description**, which assumed the z-image convention (penultimate layer,
no final norm). Mage does the opposite, and the difference is not subtle — a penultimate-layer
port is wrong by a full RMSNorm plus one decoder layer.

| step | evidence |
| --- | --- |
| the encoder runs in `embedding` mode and skips `lm_head` | `models/modules/text_encoder.py:83-84` (`_output_mode = OUTPUT_MODE_EMBEDDING`, `_skip_lm_head = True`) |
| the patched text model applies the final norm as its LAST op | `models/modules/text_encoder.py:290` — `hidden_states = self.norm(hidden_states)` immediately before `return BaseModelOutputWithPast(last_hidden_state=hidden_states, ...)` (`:292-295`) |
| the wrapper takes that tensor unchanged | `models/modules/text_encoder.py:156` (`hidden_states = outputs[0]`) → `:172-178` returns it as `last_hidden_state` |
| `TextEncoder.forward` reads `last_hidden_state` | `models/modules/text_encoder.py:535-541` |
| intermediate layers are not even materialised | `models/modules/text_encoder.py:521` passes `output_hidden_states: False` — a penultimate-layer conditioning is *unreachable* on this path |
| the system-prompt prefix is dropped per sequence | `models/modules/text_encoder.py:560` — `h_valid = h[drop_idx:]` |
| `drop_idx` = 34 (gen) / 64 (edit) | `models/utils.py:55` (`"mage-flow"` → `start_idx: 34`), `models/utils.py:64` (`"mage-flow-edit"` → `start_idx: 64`) |
| …and always arrives as an override, never from the encoder's own field | `TextEncoder` is built with `prompt_template=None` (`models/mage_flow.py:257`), so its `prompt_template_encode_start_idx` is 0; the pipeline passes `drop_idx_override` from `PROMPT_TEMPLATE` (`pipeline.py:234` and `:416`, sourced at `:275` / `:450`) |

`txt` is the concatenation of the post-drop slices; the pooled `vec` is the mean of the same
slice (`models/modules/text_encoder.py:565`) **but the DiT throws it away** —
`models/mage_flow.py:116` overwrites it with `torch.zeros(...)` before `temb = temb + txt_vec`
(`:118`). A port does not need a pooled text vector at all.

Conditioning enters the DiT as `RMSNorm(2560, eps=1e-6)` → `Linear(2560 → 3072)`
(`models/mage_flow.py:74-75`, applied at `:110` and `:115`).

**Golden:** `mage_flow_te_golden.safetensors` — `gen_hidden_full` is the pre-drop sequence and
`gen_txt` the post-drop one (likewise `edit_*`), so a port can tell "wrong layer / missing final
norm" apart from "wrong `drop_idx`". Observed: 54 gen tokens → 20 conditioning tokens (54 − 34);
157 edit tokens → 93 (157 − 64).

**Verified, not just cited.** Hooking `language_model.layers[-1]` and comparing all three
candidates against the committed `gen_hidden_full` (same host, same device, bf16):

| candidate | max abs diff vs golden |
| --- | --- |
| `penultimate` — input to `layers[-1]` (the z-image convention the epic assumed) | 10433.29 |
| `final_prenorm` — output of `layers[-1]` | 4225.29 |
| **`final_norm` — `model.norm(final_prenorm)`** | **0.000000 (bit-exact)** |

The probe therefore discriminates — the two wrong answers are off by thousands — and the golden
is unambiguously the post-RMSNorm final layer. Re-run it after any re-vendor.

Sidebar for the port's tolerance budget: running the *same* reference on MPS instead of CPU
moves that tensor by `mean_rel ≈ 2.7e-2`. Thirty-six bf16 decoder layers accumulate real
cross-backend drift, so a TE parity gate must be a tolerance, never an equality.

---

## GAP 2 — latent scale / shift constants

**Answer: there are none. The VAE latent enters the DiT unscaled and unshifted.**

- The only transform applied to the latent is a bare `img_in = nn.Linear(128 → 3072)`
  (`models/mage_flow.py:73`, applied at `:109`). That Linear absorbs whatever scale the codec
  produces, so no external constant exists to port.
- `MageVAE.encode` returns the posterior mean (or a sample) with no normalisation
  (`models/modules/mage_vae.py:615-623`); `decode` consumes `z` directly
  (`models/modules/mage_vae.py:626-633`).
- The pipeline never scales either: t2i packs the raw noise (`pipeline.py:310`), edit packs raw
  VAE latents (`pipeline.py:502-503`), and decode is `model.vae.decode(unpack(tokens.float(), …))`
  (`pipeline.py:124`).
- `grep -rn "scaling_factor\|shift_factor\|latents_mean\|latents_std" mage_flow/` returns
  **nothing**, and `vae/config.json` in every published repo carries only `latent_channels`,
  `downsample_factor`, `sample_posterior` — no scale/shift keys (contrast FLUX / SD / Mochi).

**Golden:** `mage_flow_vae_golden.safetensors` (`enc_mean` / `enc_latent` / `dec_from_latent`)
plus `mage_flow_dit_golden.safetensors::dit_in.img`, which is the latent exactly as the DiT
receives it.

---

## GAP 3 — `axes_lens` / exact msrope coordinates for packed native resolution

**Answer: Mage has no `axes_lens` config field. `MageFlowEmbedRope` precomputes a fixed
4096-entry frequency table per axis (plus a mirrored negative table), and derives coordinates
from `img_shapes` — never from `img_ids`.**

- Table construction: `pos_index = torch.arange(4096)` and
  `neg_index = torch.arange(4096).flip(0) * -1 - 1` (`models/modules/mage_layers.py:110-111`),
  each run through `rope_params(index, axes_dim[i], theta)` and concatenated
  (`:112-127`); `rope_params` is `theta**(-2i/dim)` outer-producted with the index and mapped
  through `torch.polar` to complex (`:133-144`).
- Instantiated as `MageFlowEmbedRope(theta=10000, axes_dim=axes_dim, scale_rope=True)`
  (`models/mage_flow.py:72`) — `theta` is **hardcoded in code**, not read from config (see the
  config-strip note below). `assert sum(axes_dim) == attention_head_dim`
  (`models/mage_flow.py:70`): `[16, 56, 56]` sums to 128, half-dims `[8, 28, 28]`.
- Per-segment coordinates (`models/modules/mage_layers.py:187-209`):
  - **frame** — `freqs_pos[0][idx : idx + frame]` where `idx` is the segment's position in
    `img_shapes` (`:171-177`, `:192`). For t2i there is one segment, so frame index 0.
  - **height / width** — **centred** because `scale_rope=True`:
    `cat([freqs_neg[1][-(h - h//2):], freqs_pos[1][:h//2]])` (`:194-198`) and the same for width
    (`:199-203`). So a length-`L` axis uses indices `-(L - L//2) … L//2 - 1`, not `0 … L-1`.
- Applied to **image q/k only** (`models/modules/mage_layers.py:421-422`); the text stream is
  never rotated (matching `apply_text_rotary_emb: false` in the published config). The
  convention is adjacent-pair complex: `view_as_complex(x.float().reshape(..., -1, 2))`
  (`:15-21`).
- **RoPE is built from `img_shapes`, not `img_ids`.** `MageFlow.forward` calls
  `self.pos_embed(img_shapes, device=...)` (`models/mage_flow.py:107`), and the pipeline's
  `transformer(...)` call passes only `img`, `txt`, `timesteps`, `img_shapes`,
  `img_cu_seqlens`, `txt_cu_seqlens` (`pipeline.py:190-191`). `img_ids` is still computed
  (`pipeline.py:311-314`, `:522`) and carried in the pack context (`:146`), but it never reaches
  the model — it is **vestigial**. A port should not implement it.

### Trap found here: `batch_cfg` shifts the unconditional branch's FRAME index

`_build_pack_ctx` fuses the CFG passes by duplicating the image stream and setting
`"d_img_shapes": [img_shapes[0] + img_shapes[0]]` (`pipeline.py:167`) — it **concatenates the
segment list**. `MageFlowEmbedRope.forward` derives the frame index from each segment's
`enumerate` position (`models/modules/mage_layers.py:171`, `:192`), so the duplicated
(unconditional) copy is rotated at **frame index 1**, not 0.

Driving the vendored `MageFlowEmbedRope` directly at a 16×16 latent:

| comparison | result |
| --- | --- |
| `doubled[:256]` (cond half) vs the un-doubled table | **identical** |
| `doubled[256:]` (uncond half) vs the un-doubled table | **differs, max_abs 0.9589** |
| …restricted to the frame slots `[:8]` | max_abs **0.9589** |
| …restricted to the h/w slots `[8:64]` | max_abs **0.0** |

So the difference is entirely in the frame axis, exactly as the enumerate-position derivation
predicts. Two consequences the port must not get wrong:

1. **`pipeline.py:136-140`'s claim that batch_cfg is "numerically identical to two separate
   forwards" is true for attention isolation (`cu_seqlens`) but NOT for RoPE.** `batch_cfg=True`
   and `batch_cfg=False` produce different images.
2. A port that implements the fused CFG path must replicate the frame-index shift, or run the
   two branches separately and accept that it is matching the `batch_cfg=False` trajectory.
   Pick one deliberately and pin it in the parity test — the reference default is `True`.

`mage_flow_dit_block_golden.safetensors::block_in.image_rotary_emb_{re,im}` carries the doubled
table, and `tools/verify_mage_flow_golden.py` asserts this exact split.

**Effective capacity** (the practical stand-in for `axes_lens`): 4096 indices per axis, so the
frame index must stay `< 4096`, and a spatial axis of latent length `L` needs `L//2 ≤ 4096` and
`L - L//2 ≤ 4096`, i.e. `L ≤ 8192` latent rows/cols — 131072 px at the 16× downsample, far above
the 2048-per-side native-resolution ceiling. Native-res packing therefore needs no `axes_lens`
tuning at all; the table is built once at construction.

**Golden:** `mage_flow_dit_golden.safetensors::img_shapes` (and the edit golden's, which carries
the multi-segment frame indices).

---

## GAP 4 — the four Turbo timesteps (and τ_ca / τ_dm)

**Answer: there is no separate distilled timestep table. Turbo is the identical scheduler and
the identical formula at `steps=4`.**

- `build_scheduler` (`pipeline.py:37-50`) constructs
  `FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=6.0,
  use_dynamic_shifting=False)` and calls `set_timesteps(sigmas=linspace(1.0, 1.0/N, N))`
  (`:48-49`). The scheduler applies the static shift `σ' = 6σ / (1 + 5σ)` and appends a terminal
  `0`.
- `_get_scheduler` (`pipeline.py:53-61`) prefers the repo's own scheduler — loaded from
  `scheduler/scheduler_config.json` at `pipeline.py:760-761` — and re-times it with the same
  `linspace`. That config is byte-identical (**same SHA-256**) across the four repos checked
  here — Mage-Flow (RL), Base, Edit, Turbo — as is `transformer/config.json`; Edit-Base and
  Edit-Turbo were not cached locally and are unverified.
- The denoise step is plain Euler: `img = scheduler.step(pred, t, img)` (`pipeline.py:343`),
  i.e. `x += (σ_next − σ_cur)·v`.
- **τ_ca / τ_dm are Decoupled-DMD *distillation-loss* hyperparameters, not inference state.**
  They appear nowhere in the inference repo and there is nothing to port: the distillation is
  baked into the Turbo transformer weights. Turbo differs from RL only by (a) those weights and
  (b) the README defaults `steps=4, cfg=1.0` (CFG off).

Concretely, `N=4` (Turbo):

```
sigmas    [1.0, 0.94736844, 0.85714287, 0.66666669, 0.0]
timesteps [1000.0, 947.36847, 857.14288, 666.66669]
```

(base `linspace(1, 0.25, 4) = [1, 0.75, 0.5, 0.25]` → `6s/(1+5s)` → `[1, 0.947368, 0.857143,
0.666667]`, terminal 0 appended, `t = 1000σ'`.)

**Golden:** `mage_flow_e2e_golden.safetensors` carries `sigmas_4`/`timesteps_4` (Turbo) and
`sigmas_30`/`timesteps_30` (Base) side by side.

---

## GAP 5 — exact `[τ, z_src, noisy z_tgt]` ordering and frame index for multi-image edit

**Answer: the image stream is `[noisy_target, ref_1, …, ref_N]` — target FIRST. τ is not in the
image sequence at all; it is the separate text stream.**

**This corrects the epic description's `[τ, z_src, noisy z_tgt]`** on both counts.

| fact | evidence |
| --- | --- |
| per-step assembly puts the target before its refs | `pipeline.py:552-555` — `parts.append(targets[k]); parts.append(refs[k])` then `img = torch.cat(parts, dim=1)` |
| refs are **clean** VAE latents, re-concatenated every step | same lines; the comment on `:555` reads `[1, sum(Lt+Lr), C], ref clean` |
| only target tokens are stepped | `pipeline.py:557` `pred_t = vel[:, target_idx, :]`, `:559` `scheduler.step(pred_t, t, tgt_packed)`; `target_idx` is built at `:520` |
| frame index: target 0, ref_j = j | `pipeline.py:517-518` — `shape_seq.append((1, gh, gw))  # target frame idx 0` then `shape_seq.extend(s[0] for s in ref_shapes)  # ref_j frame idx j` |
| …consumed positionally by the RoPE | `models/modules/mage_layers.py:171` (`for idx, fhw in enumerate(video_fhw)`) → `:192` (`freqs_pos[0][idx : idx + frame]`) |
| position ids are concatenated in the same order | `pipeline.py:516` — `torch.cat([tgt_ids, ref_ids], dim=1)` |
| τ (Qwen3-VL multimodal embedding, incl. the source image via the VL vision tower) is the **text** stream | `pipeline.py:538-546` builds `txt`/`neg_txt` from `_encode_edits_packed`, handed to `_build_pack_ctx` (`:548`) and passed as the separate `txt=` argument (`:190`) |
| the edit prompt body | `pipeline.py:387-393` — `"Image 1: <\|vision_start\|><\|image_pad\|><\|vision_end\|>Image 2: …{instruction}"`, templated by `mage-flow-edit` (`models/utils.py:57-65`, `drop_idx` 64) |
| refs are VAE-encoded at the **target** resolution | `pipeline.py:501-502` (`_preprocess_ref_image(p, h_, w_, dev)` then `compute_vae_encodings`) |
| but the VL conditioning image is long-edge capped at 384 | `pipeline.py:533` via `_resize_long_edge`, default `vl_cond_long_edge=384` (`:425`) |

One extra trap worth carrying into the port: **the edit path SAMPLES the VAE posterior.**
`ModelConfig.vae_sample_posterior` defaults to `True` (`models/mage_flow.py:35`) and
`load_from_repo` never overrides it, so the published `vae/config.json`'s
`sample_posterior: false` is **not** what the pipeline uses; sampling is seeded through the
global RNG (`torch.manual_seed(seeds[i])`, `pipeline.py:499`). No port can reproduce torch's
global RNG bit-for-bit, so the edit golden gates the posterior *moments*
(`mage_flow_vae_golden.safetensors::enc_mean` / `enc_logvar`) and the port applies its own RNG.
The watermark-detection entry point is documented to take the deterministic mean instead (`pipeline.py:576-594`, `invert_to_noise`).

**Golden:** `mage_flow_edit_golden.safetensors` — `seq_step0` is the literal assembled stream
(`[noisy_target(gh·gw), ref_1(gh·gw), …]`, doubled by `batch_cfg` with the cond half first) and
`img_shapes` is the per-segment `(frame, h, w)` table.

---

## GAP 6 — main-training timestep sampling distribution

**UNRESOLVED — and not resolvable from this source. `microsoft/Mage` ships inference only.**

There is no trainer, loss, optimizer, or timestep sampler anywhere in the vendored tree
(`grep -rn "def train\|loss_fn\|optimizer\|logit_normal\|timestep.*sampl" mage_flow/` returns
nothing). `ModelConfig` carries a couple of training-oriented flags (`vae_encoder_only`,
`compile_vae_encoder`, `models/mage_flow.py:36-37`) and `MageFlow.forward` has a
`self.training and self.checkpoint` gradient-checkpointing branch (`models/mage_flow.py:123-133`),
but no sampling policy.

The only timestep distribution stated in the paper is `t ~ U(0, 1)` for **Mage-VAE Stage-I**,
which is the codec's own pretraining objective — not the DiT's flow-matching schedule.

**Recommendation for P5 (sc-14054 / sc-14055 / sc-14056):** adopt the `mlx-gen-z-image` trainer's
default timestep distribution (the same DiT lineage) and **record the choice explicitly on the
training story**, so it reads as a deliberate substitution rather than a discovered fact. If
Microsoft later publishes the training code, this is the one gap to revisit.

---

## Other pins confirmed while reading (not part of the six, but load-bearing)

- **The initial latent is Gaussian-Shading watermarked noise, not `randn`.**
  `pipeline.py:303-308` computes `get_noise(...)` (plain seeded `randn`) and immediately
  **overwrites** it with `encode_noise(shape, key=gs_key_int, seed=seeds[i])`; same at
  `:504-507` for edit. Payload `"MageFlow"`, 256 bits, default key `20260720`
  (`models/modules/mage_latent.py:10,16,19`), built as the inverse normal CDF of key-seeded
  watermarked uniforms (`:76-90`). Detection is `invert_to_noise` (`pipeline.py:576-629`) +
  `decode_bits` (`models/modules/mage_latent.py:93-119`). A port that seeds a plain `randn` is
  wrong from token 0 — `mage_flow_noise_golden.safetensors` dumps both tensors so the test can
  assert a match against one and a mismatch against the other.
- **Content screening is mandatory and fail-closed.** `TextEncoder.screen_text` /
  `screen_edit` (`models/modules/text_encoder.py:590`, `:637`) run a Qwen3-VL `.generate()`
  JSON-verdict pass on every prompt, with no opt-out; any error yields `violates=True` and a
  blank white refusal image (`models/modules/mage_text.py:250-259`). SceneWorks will use its own
  moderation — that divergence needs its own decision story.
- **The DiT MLP is `gelu-approximate`, not SwiGLU** (`models/modules/mage_layers.py:547`, `:557`),
  on both streams. This is a real difference from the z-image sibling the port reuses.
- **The timestep frequency table is deliberately downcast to bf16**
  (`models/modules/mage_layers.py:45`) — the model was trained with that rounding, so an fp32
  table is wrong. The embedder is `Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=0,
  scale=1000)` → `TimestepEmbedding(256 → 3072)` (`:93-94`), fed the scheduler **sigma ∈ [0,1]**
  directly (`pipeline.py:189`).
- **The DiT reads only nine config fields.** `load_from_repo` strips everything else before
  building `MageFlowParams` (`pipeline.py:729-737`): only `in_channels`, `out_channels`,
  `context_in_dim`, `hidden_size`, `num_heads`, `depth`, `axes_dim`, `checkpoint`, `patch_size`
  survive. `mlp_ratio`, `theta`, `rope_type`, `static_shift`, `schedule_mode`, … are hardcoded in
  code, so the published config values for them are documentation, not configuration.
- **Joint attention order is `[text, image]`** with QK-RMSNorm on both streams and
  `causal=False` (`models/modules/mage_layers.py:424`, `:490`).
