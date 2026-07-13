# Full Codebase Review — mlx-gen — 2026-07-11

## Executive summary

- **Repository at a glance:** Rust workspace (Apple-MLX on-device inference of generative image/video models), 33 packages (30 `mlx-gen-<family>` provider crates + `mlx-gen` core + `gen-core` + `gen-core-testkit`), ~976 Rust files, ~265k LOC.
- **Coverage:** Whole workspace, every crate, via 17 subsystem reviewers + 3 cross-cutting sweeps (sequential-residency pattern, CI/supply-chain/docs, prior-findings verification), all six lenses (security, bad-pattern, redundant, dead-code, efficiency, readability). Excluded: `_vendor/` (gitignored third-party checkouts), `tools/golden/` (gitignored regenerable artifacts), `target/`. All three High findings were re-verified against source by the coordinator.
- **Headline:** This is the 4th whole-workspace review. The standout result is remediation: **all 28 tracked findings from the 2026-07-01 review (all 6 Highs + 22 selected Mediums) are verified FIXED in current source**, each with the actual guard in code and usually a regression test (epic 9084, PRs #631–#647, plus sc-10087/sc-9500 follow-ons). Zero exploitable security issues, zero `unsafe`, zero Criticals. The new risk is concentrated in the two big post-07-01 waves: the **sequential-residency rollout (epic 10834)** shipped as an 8-way copy-pasted scaffold with systematic cancel/`clear_cache` gaps and a silent partial rollout (High), and the **new/changed provider code** (boogu img2img, anima, krea edit/control) re-instantiated known bug classes — a default path routed through a solver the crate's own survey rejects (High), and an advertised-but-inert scheduler axis (High). The dominant systemic theme remains "fixes don't travel": roughly a third of all findings are a guard/fix that exists in one crate (often in the same file) missing from a sibling.
- **Counts:** Critical: 0 | High: 3 | Medium: 48 | Low: 89 | Info: 41 — **181 findings** (F-001…F-181, numbered in discovery order; cross-crate duplicates consolidated).

### Status of the 2026-07-01 findings

All 28 tracked items verified fixed against current source: F-001/002 (mask buffer guards + tests), F-003 (SAM3 box validation), F-004 (SD3 empty-prompt BOS, with new e2e golden), F-005 (Kolors CFG-off B=1 conditioning + contract tests), F-006/F-013 (PiD cancel + budget/tiling), F-007/F-008 (steps≥1 floor, typed `Unsupported`), F-009 (descriptor conformance sweep across 23 crates), F-014–F-019 (wan/ltx/lens cancel + trainer window math), F-027 (flux2 ref cap), F-033 (krea seed), F-034/F-035 (SD3 shift flag + variant preflight), F-036–F-038 (sensenova/bernini), F-040/F-041 (SAM state/eviction), F-045 (converter dedup incl. LICENSE + quant marker), F-076 (packed-tier mismatch rejection), F-079 (krea prepare hoist). Supply chain: F-046/F-047/F-049/`--locked` fixed; F-048 (core-llm `branch = "main"`) deliberately retained with a corrected rationale and frozen by `Cargo.lock` + CI `--locked` — the true fix is gated on an upstream mlx-llm re-pin. F-050 (CLAUDE.md) was fixed and has **re-drifted** (see F-081).

---

## High findings

#### [F-093] Give Boogu Turbo img2img a native DMD branch — the default request silently renders through the excluded Euler solver
- **Category:** bad-pattern
- **Severity:** High
- **Location:** `mlx-gen-boogu/src/pipeline.rs:448-506` (turbo img2img), `mlx-gen-boogu/src/pipeline.rs:383-408` (t2i scheduler-only branch); contrast `mlx-gen-boogu/src/model.rs:102-126` (`TURBO_SAMPLERS` rationale)
- **Finding:** `generate_turbo_img2img_with_progress` (new since 07-01, sc-10191) has no native-DMD branch: it always calls `run_flow_sampler(opts.sampler.as_deref(), …)`, and the curated runner maps `None` → `Euler` (`src/sampler.rs:332-334`). The same happens on t2i when a request sets only a `scheduler`. The crate's own sc-7491 survey documents deterministic Euler as out-of-regime for this DMD student ("background artifacts") and deliberately excludes it from `TURBO_SAMPLERS`.
- **Impact:** Every default-configured Turbo img2img render (and any turbo request selecting a scheduler without a sampler) is produced by a solver the descriptor refuses to advertise — silent quality divergence on a brand-new default path, inconsistent with t2i (native DMD loop by default).
- **Suggested fix:** Add a native DMD predict→renoise branch seeded from the blended latent (mirroring `generate_turbo_with_progress`'s split); when only a scheduler is set, default the sampler to `lcm` (the surveyed closest match) or keep the native loop over the re-shaped grid — never fall through to Euler on Turbo.
- **Confidence:** High (coordinator re-verified the routing)

#### [F-115] Wire `req.scheduler` into Anima's sigma schedule or stop advertising it
- **Category:** bad-pattern
- **Severity:** High
- **Location:** `mlx-gen-anima/src/model.rs:42-43,154-161`; `mlx-gen-anima/src/pipeline.rs:61-69,233`
- **Finding:** The descriptor advertises `schedulers: curated_scheduler_names()`, so the shared floor accepts any curated name — but `GenOptions` has no scheduler field, `generate_impl` drops `req.scheduler`, and `denoise_loop` always uses the fixed `anima_sigmas(steps)`. Every flow sibling threads `req.scheduler` through `resolve_flow_schedule`; anima is the only flow provider that validates the name and then ignores it.
- **Impact:** An advertised capability that has never worked: a worker requesting any scheduler gets the native linspace-shift3 schedule with no error and no event — the same "capability advertised but inert" class as the 07-01 Kolors CFG-off finding, and a break of the epic-7114 per-generator scheduler axis contract.
- **Suggested fix:** Thread `req.scheduler` through `GenOptions` and resolve via `resolve_flow_schedule(req.scheduler.as_deref(), SIGMA_SHIFT.ln(), steps, &anima_sigmas(steps))`, or shrink `capabilities.schedulers` to only the native entry so the floor rejects the rest.
- **Confidence:** High (coordinator re-verified)

#### [F-172] Wire Sequential residency across the whole z-image family, not just the Turbo flagship
- **Category:** bad-pattern
- **Severity:** High
- **Location:** `mlx-gen-z-image/src/model.rs:174-197` (wired) vs `mlx-gen-z-image/src/model_base.rs:116-135`, `mlx-gen-z-image/src/model_control.rs:90`, `mlx-gen-z-image/src/model_base_control.rs:106-115` (not wired)
- **Finding:** sc-10839 wired `OffloadPolicy::Sequential` only into `z_image_turbo`. The three sibling generators registered by the same crate (`z_image`, `z_image_turbo_control`, `z_image_control`) never read `spec.offload_policy` — `model_base.rs` even calls the shared `load_components` that was factored into exactly the per-phase loaders Sequential needs. Later stories show family-wide coverage is the intended bar (sc-11006 qwen edit+control, sc-11101 "the whole krea_2 family").
- **Impact:** A caller (the SceneWorks fit-gate) that selects Sequential to fit a small machine gets a silent full-resident load on the z-image base/control variants — the exact OOM the feature exists to prevent, with no error and no way to detect it (see F-176). The "fixes don't travel to siblings" failure mode recurring inside a single crate.
- **Suggested fix:** Route the three variants through the same `Residency` enum; `load_text_encoder_only`/`load_heavy` already exist and are shared. The control variants need a control-branch analog of qwen's `model_control.rs` split.
- **Confidence:** High (coordinator re-verified: `offload_policy` is read only in `model.rs`)

---

## Medium findings

### gen-core contract layer

#### [F-001] Extend the F-053 finiteness floor to the new guidance/conditioning knobs
- **Category:** security
- **Severity:** Medium
- **Location:** `gen-core/src/generator.rs:547-647`
- **Finding:** `Capabilities::validate_request` rejects non-finite `guidance`/`true_cfg` (the F-053 fix), but the epic-7434 fields added since — `guidance_eta`, `guidance_momentum`, `guidance_norm_threshold` — plus `strength`, `control_scale`, `scheduler_shift`, `image_guidance`, and `Conditioning::Control { scale }` accept NaN/±Inf. A NaN eta flows into `guidance::normalize_diff`; a NaN momentum permanently poisons the `MomentumBuffer` running state.
- **Impact:** A request-supplied NaN silently NaN-poisons the whole denoise (garbage output, no error); for the momentum buffer it persists across every remaining step. Engines do not re-check finiteness — the floor is documented as the shared guard.
- **Suggested fix:** Add the same `is_finite()` rejection for every `Option<f32>` knob the floor owns, including `Control.scale` inside the conditioning loop.
- **Confidence:** High

#### [F-002] Type capability-gap rejections outside the generator floor as `Error::Unsupported`
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `gen-core/src/caption.rs:194-226`, `gen-core/src/control.rs:102-133`
- **Finding:** The F-008 remediation typed the *generator* floor's capability gaps, but `CaptionCapabilities::validate_request` and `ControlBranch::resolve_control` still return `Error::Msg` for capability gaps (unsupported caption_type/length, "supports pose control only"). `error.rs` states candle gating "depends on this being typed".
- **Impact:** Consumers distinguishing "backend can't do that" from "malformed request" get the wrong variant for captioner and control-branch gaps.
- **Suggested fix:** Switch the capability-gap branches (not the malformed-value ones) in both files to `Error::Unsupported`.
- **Confidence:** High

### Core `src/` (adapter loader)

#### [F-010] Extend the sc-10578/sc-10678 packed-tier deferral and memory guard to the BFL fused→split LyCORIS path
- **Category:** efficiency
- **Severity:** Medium
- **Location:** `src/adapters/loader.rs:1303-1362` (`install_bfl_lycoris`), vs the guarded twin at `src/adapters/loader.rs:292-403`
- **Finding:** `install_bfl_lycoris` — the install path for every BFL/ComfyUI-named LoKr/LoHa on a FLUX.1/FLUX.2 host — always materializes the full fused `[Σout, in]` bf16 delta with no `is_quantized()` branch, no `LokrStructured` deferral, and none of the sc-10678 `materialization_exceeds_budget` pre-flight its plain-path twin gained. On the default Q8 tier a full-coverage BFL LoKr on FLUX holds multiple GB of resident deltas with no up-front refusal.
- **Impact:** A BFL-named adapter that would be budget-refused (or deferred allocation-free) on the plain path instead risks an uncatchable mid-load OOM/SIGKILL of the worker on smaller Macs — the exact failure sc-10678 prevents.
- **Suggested fix:** Sum `projected_delta_bytes` over the resolved destinations and apply the same `materialization_exceeds_budget` refusal before reconstructing; document the no-deferral scope next to the sc-10678 comment.
- **Confidence:** High

### SD3 / SANA

#### [F-018] Adopt the shared `Capabilities::validate_request` in SD3
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `mlx-gen-sd3/src/model.rs:289-351`
- **Finding:** SD3's hand-rolled validator never calls the shared floor (which sibling SANA uses). Missed: the F-053 non-finite-guidance guard (`guidance: Some(NAN)` NaN-poisons every step), sampler/scheduler membership (unknown names silently fall back to the native path), and rejection of `true_cfg`/`guidance_method` (silently dropped knobs).
- **Impact:** Guards centralized in gen-core are bypassed on all three SD3 ids; NaN guidance renders garbage instead of a typed refusal; misspelled sampler names silently produce wrong-but-plausible output.
- **Suggested fix:** Call `desc.capabilities.validate_request(id, req)?` first (as SANA does), then keep only the SD3-specific extras.
- **Confidence:** High

#### [F-019] Gate SD3 CFG on `guidance > 1.0` to match the diffusers SD3 reference
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `mlx-gen-sd3/src/model.rs:207`; `mlx-gen-sd3/src/pipeline.rs:124-135`
- **Finding:** `cfg_on = guidance != 1.0` runs the uncond branch for any `0 < guidance < 1` (or ≤ 0). SD3 uses the un-shifted diffusers combine, and diffusers gates `do_classifier_free_guidance = guidance_scale > 1`; the z-image `!= 1.0` rationale applies only to the shifted `scale = guidance − 1` convention, which is not SD3's. SANA already uses `> 1.0`.
- **Impact:** Sub-1 guidance on `sd3_5_large`/`sd3_5_medium` diverges from the reference output and doubles per-step cost.
- **Suggested fix:** Change both gates to `guidance > 1.0` (keeping the F-094a validation).
- **Confidence:** Medium — verify against the diffusers SD3 pipeline before changing.

#### [F-020] Hoist SANA's per-image prompt/reference encoding out of the `count` loop
- **Category:** efficiency
- **Severity:** Medium
- **Location:** `mlx-gen-sana/src/model.rs:352-373`; `mlx-gen-sana/src/pipeline.rs:488-526, 583-607`
- **Finding:** `generate_impl` calls `pipeline.generate_with` once per image; each call re-runs the full Gemma-2-2B prompt encode (and negative encode under CFG) plus the DC-AE init encode — all seed-independent. SD3 already encodes once outside its count loop.
- **Impact:** A `count = 8` batch pays 8× (16× with CFG) the most expensive non-denoise component.
- **Suggested fix:** Split encode/denoise (or accept pre-encoded conditioning) and hoist the text + init encodes to the per-request preamble.
- **Confidence:** High

### Lens

#### [F-029] Restore an effective cancel in the 20B MoE encode — the F-019 fix is now graph-time-only (false green)
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `mlx-gen-lens/src/text_encoder/encoder.rs:149-177`; `mlx-gen-lens/src/pipeline.rs:259-277`
- **Finding:** The F-019 remediation's premise ("routing forces a host sync per layer") went stale when sc-9500 moved MoE routing on-device: nothing in `encode` calls `eval`/`item`, so all ~24 per-layer cancel checks execute in microseconds during lazy graph *construction*; the actual 20B×2-prompt compute happens later in one uninterruptible `eval`.
- **Impact:** A cancel issued during the dominant non-denoise stage is not honored until the entire encode completes — the exact gap F-019 reported, now hidden behind checks that look like a fix.
- **Suggested fix:** Force materialization at the cancel checkpoints (e.g. `hidden.eval()?` per layer, or at the 4 captured layers) when `cancel.is_some()`; update the stale doc comment.
- **Confidence:** High

#### [F-030] Fix the lens step-progress total and Decoding emission under PiD early-stop
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `mlx-gen-lens/src/registry.rs:336-340, 361-409`
- **Finding:** `generate_impl` sets `total = steps` and emits `Progress::Decoding` only when `cur >= total`, but PiD `from_ldm` early-stop truncates the schedule to `keep` entries, so the emitted `Step` never reaches its own `total` and `Decoding` is never emitted — precisely on the path whose 4×-SR decode is the longest.
- **Impact:** The job appears frozen at `(keep−1)/steps` and the decode is invisible.
- **Suggested fix:** Compute the effective step count from the resolved sigmas and use it as both the emitted `total` and the Decoding trigger.
- **Confidence:** High

### FLUX.2

#### [F-035] Cap the caption-upsample prompt at a token budget — the prefill is unbounded
- **Category:** security
- **Severity:** Medium
- **Location:** `mlx-gen-flux2/src/caption_upsample.rs:149-173`; `mlx-gen-flux2/src/model.rs:442-474`
- **Finding:** The F-012 remediation capped the upsample **decode** length (`MAX_NEW_TOKENS_CAP = 2048`), but the **prefill** has no bound: `build_upsample_input_ids` tokenizes with no truncation (unlike the T2I encode, which right-truncates at 512), then embeds `[1, n, 5120]` f32 and runs one uncancellable large-decoder prefill with O(n²) causal SDPA.
- **Impact:** A multi-megabyte prompt with `enhance_prompt: true` on `flux2_dev`/`flux2_dev_edit` validates cleanly and produces an arbitrarily large allocation/compute burst — the repo's historical highest-severity class (request-input OOM); MLX allocation failure is not a recoverable `Err`.
- **Suggested fix:** Truncate the rendered upsample input at a token budget (a few × `MAX_LENGTH`) before `encoder.embed`, or reject oversize prompts in `validate_request`.
- **Confidence:** High

### SAM3

#### [F-043] Validate point/label counts in the SAM3 tracker point-prompt path
- **Category:** security
- **Severity:** Medium
- **Location:** `mlx-gen-sam3/src/tracker.rs:431-465` (`PromptEncoder::encode_points`), reachable from `segment_points` (`tracker.rs:1869-1877`)
- **Finding:** `encode_points` sizes the positional-embedding tensor from `points.len()` but iterates `labels`; the documented `labels.len() == points.len()` invariant is never checked. With more labels than points, `take_axis` gathers past the end of the embedding tensor (unchecked OOB GPU gather); with fewer, trailing clicks are silently dropped. The sam2 sibling validates the same mismatch; the sam3 box path was hardened in F-003 — the point path was missed.
- **Impact:** A malformed point prompt on the interactive smart-select path produces undefined GPU reads / silently corrupt masks instead of a typed rejection.
- **Suggested fix:** Reject `labels.len() != points.len()` at the top of `encode_points` (mirroring sam2's guard); optionally reject non-finite coordinates.
- **Confidence:** High

### LTX

#### [F-050] Fix progress `current` overrunning `total` under multi-eval curated samplers
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `mlx-gen-ltx/src/model.rs:579-586`; `mlx-gen-ltx/src/pipeline.rs:941-945, 988-992`
- **Finding:** LTX's `on_step` closure ignores the σ-derived monotone value it is passed and blindly increments its own counter. The 2nd-order solvers LTX advertises (`heun`, `dpmpp_sde`) evaluate twice per step, so a heun T2V run reports `current` up to 22 against `total: 11`.
- **Impact:** Progress contract violation — consumers see >100% progress for any request selecting a multi-eval sampler (user-reachable via `req.sampler`).
- **Suggested fix:** Dedupe on the forwarded `current` (only advance when it increases) or pass the σ-derived value through with a per-stage offset.
- **Confidence:** High

#### [F-051] Add cancellation to LTX's tiled VAE decode, conditioning encodes, and audio decode
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `mlx-gen-ltx/src/vae.rs:616-687`; `mlx-gen-ltx/src/pipeline.rs:163-174, 1109-1138`; `mlx-gen-ltx/src/model.rs:685-787`
- **Finding:** The sc-9093 cancel remediation reached LTX **enhance** only. `decode_tiled` runs a t×h×w tile loop with a per-tile `eval` but no `CancelFlag`; `decode_audio_track` and the per-keyframe/per-clip VAE encodes (unbounded keyframe/clip count) never check `req.cancel`.
- **Impact:** At the request ceiling (1280×1280×1025 frames) the tiled decode is a minutes-long uncancellable stage after `Progress::Decoding` — the known workspace class (07-01 theme T3) re-instantiated in seams the remediation didn't reach.
- **Suggested fix:** Thread `&CancelFlag` into `decode_tiled` (per-tile check where the `eval` already syncs), `decode_to_frames`, `decode_audio_track`, and the conditioning-encode loops; cap the accepted keyframe/clip count in `validate_request`.
- **Confidence:** High

### Wan

#### [F-058] Guard zero-rank LoRA factors in the wan fold-path merge like the additive path already does
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `mlx-gen-wan/src/adapters.rs:334-349` (vs the guarded twin at `:775-778`)
- **Finding:** The new additive installer rejects rank-0 factors; the pre-existing fold path `merge_one` does not: a LoRA with a 0-size `lora_A` leading dim yields `alpha/rank = 0/0 = NaN` (or `∞`), and the whole base weight is silently replaced by NaN via the merge. Guard added to the new path (sc-10044) but not back-ported 400 lines up in the same file.
- **Impact:** Silent full-weight NaN corruption from a degenerate user-supplied adapter on the dense (default) tier — every subsequent render is garbage while the load reports success; the packed tier errors cleanly for the same file.
- **Suggested fix:** Hoist the rank>0 (and 2-D shape) validation into the shared grouping step used by both `merge_one` and `install_one_lora_additive`.
- **Confidence:** High

#### [F-059] Reject empty `.alpha` tensors instead of panicking in wan's `read_alpha`
- **Category:** security
- **Severity:** Medium
- **Location:** `mlx-gen-wan/src/adapters.rs:101-105`
- **Finding:** `read_alpha` does `as_slice::<f32>()[0]`; a `[0]`-shaped `.alpha` tensor (malformed/truncated adapter file) makes the slice empty and the index panics. Both fold and additive paths call this on any `.alpha` key in a user-supplied safetensors, on all five Wan generator entries.
- **Impact:** Process abort (worker crash) from a malformed adapter file reachable through `LoadSpec.adapters`, instead of the crate's typed error.
- **Suggested fix:** `.first().copied().ok_or_else(|| Error::Msg("empty .alpha tensor".into()))`.
- **Confidence:** High

### Krea

#### [F-069] Guard (or honor) the PiD `from_ldm` early-stop on the krea edit path
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `mlx-gen-krea/src/model.rs:492-523, 572-587`; `mlx-gen-krea/src/pipeline.rs:441-511`
- **Finding:** `generate_impl` resolves `(capture_sigma, keep)` and builds the PiD decoder at σ_capture, but the edit branch calls `render_edit(...)`, which takes no `keep` and always denoises to σ=0. The sc-10121 img2img+capture conflict guard keys on `reference.is_some()`, which is `None` on the edit path — so `krea_2_edit` + `use_pid` + `pid_capture_sigma` sails through with the decoder built for a partially-denoised latent it never receives.
- **Impact:** Silent σ-desync on the decoder — the corruption mode the img2img guard exists to prevent — on a request-reachable combination.
- **Suggested fix:** Thread `keep` into `render_edit` (truncate like `render_base`), or extend the conflict guard to the edit path with a tracked-story message.
- **Confidence:** High

#### [F-070] Reject (don't silently ignore) `use_pid` / `spec.pid` on the krea pose-control lane
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `mlx-gen-krea/src/model_control.rs:119-150, 237-287`
- **Finding:** `krea_2_turbo_control` never consults `req.use_pid` or `spec.pid`: a `use_pid: true` request succeeds with a native-VAE (non-super-resolved) image, and a `spec.pid` load is silently inert. The base krea lanes error loudly; the qwen control sibling fully wires PiD. The pipeline doc says the seam is "intentionally NOT wired" — intent should surface as a validation error, not a silent downgrade.
- **Impact:** A worker requesting PiD decode gets a quietly different result; the load-spec `pid` field is dead on arrival — the false-green/silent-descope class.
- **Suggested fix:** Reject `spec.pid` at load and `req.use_pid` at validate with a message naming the deferral, matching the base lane's failure mode.
- **Confidence:** High

#### [F-071] Ground the Qwen3-VL context on BOTH krea edit sources — the person image is invisible to the grounded encode
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `mlx-gen-krea/src/model.rs:533-540`; `mlx-gen-krea/src/pipeline.rs:818-824`; `mlx-gen-krea/src/text_encoder/encoder.rs:176-181`
- **Finding:** The scene+person `MultiReference` edit (epic 10871 P1.3) VAE-encodes both sources as in-context tokens, but the grounded half runs on `sources[0]` only — `encode_grounded` takes a single reference and `forward_with_image` splices only the first `<|image_pad|>` run, while the tokenizer/`image_token_runs`/`mrope_positions` already handle N images. The comments concede this without a visible tracking story.
- **Impact:** The person image never reaches the Qwen3-VL grounding the edit LoRA conditions on — identity-transfer quality on two-source edits diverges from the reference dual conditioning; a quiet capability gap on the advertised `MultiReference` surface.
- **Suggested fix:** Extend `encode_grounded` to `&[&Image]` (the helpers are shaped for it); if genuinely deferred, surface as a validation error or tracked story, not a comment.
- **Confidence:** High that grounding is first-source-only; Medium on quality impact vs the reference.

#### [F-072] Load the krea Qwen3-VL vision tower lazily / edit-only instead of eagerly for every variant
- **Category:** efficiency
- **Severity:** Medium
- **Location:** `mlx-gen-krea/src/pipeline.rs:68-87`; `mlx-gen-krea/src/loader.rs:32-45`
- **Finding:** `KreaText::from_snapshot` eagerly builds the f32 vision tower for all variants — Turbo/Raw t2i, img2img, and pose-control never call `encode_grounded`, yet pay the tower's load + residency (~0.6 GB) on every load, and again on **every generate** under Sequential. It also makes `visual.*` a hard load dependency for plain t2i. The comment self-flags this as "tracked for P3" but it shipped as the family default.
- **Impact:** Wasted unified memory and Sequential per-job load time on the dominant paths; a latent compat break for vision-less snapshots.
- **Suggested fix:** Make `vision` an `Option<VisionTower>` loaded on first grounded encode, or an edit-only constructor keyed off the descriptor id.
- **Confidence:** High

#### [F-073] Hoist krea's step-invariant per-request work out of the count loop and CFG branches
- **Category:** efficiency
- **Severity:** Medium
- **Location:** `mlx-gen-krea/src/model.rs:561-639`; `mlx-gen-krea/src/pipeline.rs:219-221, 264-267, 383-386, 459-476`
- **Finding:** Per image in the count loop, krea re-runs: the img2img/edit/control VAE encodes of the same source images (an 8-count two-source edit performs 16 identical full-res VAE encodes), `prepare`/`prepare_edit` (text fusion + host f64 RoPE build, ~3M trig ops at 2048² edit), and — under edit CFG — the **vision tower twice** on the same source (once per pos/neg context) when only the text differs.
- **Impact:** Linear-in-count waste of the most expensive non-denoise stages; multi-image edit jobs pay seconds of pure redundancy per extra image.
- **Suggested fix:** Encode references/pose once before the count loop; build `JointPrep`/`EditPrep` once per request; split `encode_grounded` so the vision forward is shared across pos/neg.
- **Confidence:** High

### Docs / repo hygiene

#### [F-081] Keep CLAUDE.md's crate inventory and reuse map in sync — re-drift within a week of the F-050 fix
- **Category:** readability
- **Severity:** Medium
- **Location:** `CLAUDE.md:16-24`
- **Finding:** CLAUDE.md says "(29 crates)" but there are 30 family crates: `mlx-gen-anima` (added 07-09/10) is absent from the list and the reuse map. The map also omits real edges in the manifests: `anima → z-image + qwen-image`, `krea → qwen-image + boogu`, `ideogram → flux2`, `lens → flux2`, `bernini → wan`, `scail2 → wan`.
- **Impact:** CLAUDE.md is the load-bearing context for every agent session; a missing edge means an agent editing `qwen-image`/`boogu` won't know `krea`/`anima` consume its types and will lint crate-scoped instead of `--workspace` — the exact breakage the map exists to prevent.
- **Suggested fix:** Add `-anima` (count → 30), add the missing reuse edges, note anima's training surface. Consider dropping literal counts (they always rot — see F-090).
- **Confidence:** High

#### [F-082] Refresh README "Supported models" — five shipped families are missing
- **Category:** readability
- **Severity:** Medium
- **Location:** `README.md:5-19`
- **Finding:** The model list omits Krea 2 (incl. Raw LoRA training), SD3.5, SANA, Anima (incl. training), and the catalog-wide PiD decoder; the Training bullet lists only 6 families; "two dozen provider crates" undercounts 30.
- **Impact:** The public front door materially under-describes the project — the T7 doc-drift theme.
- **Suggested fix:** Add the five families, extend the Training bullet, fix the count.
- **Confidence:** High

#### [F-083] Regenerate or drop CODEGRAPH.md — it describes the wrong project
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `CODEGRAPH.md:1-17`
- **Finding:** The committed auto-generated summary (last analyzed 06-17) describes mlx-gen as "a conformance testing toolkit" — it summarized only `gen-core-testkit` and presents that as the whole repo, with "Confidence: High", and still references the removed text-LLM trait.
- **Impact:** A confidently wrong committed summary actively misleads agents and readers — worse than no file.
- **Suggested fix:** Re-run the CodeGraph analysis against the full workspace, or delete the file until the generator produces an accurate summary.
- **Confidence:** High

#### [F-084] Reconsider the four ~50MB committed SenseNova fixtures
- **Category:** efficiency
- **Severity:** Medium
- **Location:** `mlx-gen-sensenova/tests/fixtures/` (`it2i_golden` 51MB, `vqa_golden`, `t2i_golden`, `interleave_golden` ~50MB each)
- **Finding:** ~200MB of the repo's pack is four synthetic goldens — an order of magnitude above every other crate's fixtures and against the "small/synthetic" convention. Each regeneration re-adds ~200MB of non-delta binary history.
- **Impact:** Permanent clone/CI weight and growing history bloat; a precedent new crates may copy (anima's whole golden set is <200KB — the right pattern).
- **Suggested fix:** On next regeneration, shrink to the anima-style subsample+statistics form, or move to the gitignored `tools/golden/` tier. History rewrite not warranted.
- **Confidence:** Medium — whether the full tensors are load-bearing for those parity tests was not verified.

### Boogu

#### [F-094] Bound and validate Boogu Edit reference images at the Generator boundary
- **Category:** security
- **Severity:** Medium
- **Location:** `mlx-gen-boogu/src/model.rs:450-490`; `mlx-gen-boogu/src/pipeline.rs:566-580`
- **Finding:** `validate_request` checks only the ref count (1..=5). Dimensions are validated (multiple-of-16) only inside the pipeline — so `validate()` returns Ok for a request `generate()` rejects — and are never bounded: the Edit spatial path VAE-encodes each reference at its **raw** resolution and packs `(rH/16)·(rW/16)` tokens per reference into the DiT sequence.
- **Impact:** Five 8192² references pass validation, then drive unbounded VAE encodes and a quadratically exploding joint-attention sequence — request-reachable OOM/GPU abort; also violates the validate-before-work contract.
- **Suggested fix:** In `validate_request` for the Edit id, check per-reference dims and clamp to the advertised envelope (or smart-resize refs to a bounded area like the vision path does).
- **Confidence:** High

### Qwen-Image

#### [F-101] Route qwen validation through the shared floor — scheduler/guidance_method/NaN-guidance/steps-0 all slip through
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `mlx-gen-qwen-image/src/model.rs:505-563` (shared by all three generators)
- **Finding:** Qwen's hand-rolled `validate_request` never calls the shared floor: (a) `req.scheduler` is never checked — unknown names silently fall back to the native schedule while unknown *samplers* are rejected; (b) `guidance_method` is silently ignored; (c) non-finite `guidance`/`true_cfg` accepted (NaN renders garbage); (d) `steps: Some(0)` with `sampler: "lightning"` passes the carve-out and silently renders a 1-step image.
- **Impact:** Silently-wrong or silently-downgraded output for fields the descriptor advertises as validated, across all three qwen ids.
- **Suggested fix:** Call `caps.validate_request(MODEL_ID, req)?` first (as z-image does), then layer the qwen-specific checks; drop the duplicated blocks.
- **Confidence:** High

### FLUX.1 / Chroma / PuLID

#### [F-105] Bring the flux1 base bespoke `validate` back up to the shared floor
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `mlx-gen-flux/src/model.rs:484-566`
- **Finding:** `flux1_dev_control` delegates to the completed shared floor, but the base `flux1_dev`/`flux1_schnell` validator stayed bespoke (for the IP-Adapter carve-out) and never absorbed the new checks: `steps: Some(0)` silently clamps to 1; `guidance: Some(NAN)` flows into `time_text_embed` and NaN-poisons the render; `true_cfg: Some(NAN)` survives `.clamp` (Rust clamp returns NaN); `guidance_method` is silently ignored.
- **Impact:** The T2 "hand-rolled validate drifts below the floor" class re-opened for the flux variant not named in the 07-01 review.
- **Suggested fix:** Add the three floor checks, or call `caps.validate_request` first with the IP-Adapter carve-out as a pre-filter (as `model_control.rs` does).
- **Confidence:** High

#### [F-106] Make PuLID `generate` run PuLID's own validation floor — the `hyper` trap survives in the generate path
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `mlx-gen-pulid/src/pulid_flux.rs:254-304` (vs `validate` at `:225-239`)
- **Finding:** F-026 was fixed only on the `validate()` side; `generate_impl` never calls `self.validate(req)`. The only floor that runs is the FLUX-dev backbone's — and dev advertises `hyper` while PuLID deliberately omits it. A caller invoking `generate` without a prior `validate` (nothing enforces the sequence; flux and chroma both self-validate inside generate) gets an 8-step render **without** the Hyper LoRA — the documented "undertrained noise" trap.
- **Impact:** The F-026 bug class is half-remediated; sibling defense-in-depth didn't travel to pulid.
- **Suggested fix:** First line of `generate_impl`: `self.validate(req)?`.
- **Confidence:** High

#### [F-107] Chroma re-runs the full T5-XXL prompt encode (pos + neg) for every image in a batch
- **Category:** efficiency
- **Severity:** Medium
- **Location:** `mlx-gen-chroma/src/model.rs:558-578` → `denoise_with_schedule` at `:293-353`
- **Finding:** The per-count loop re-tokenizes and re-runs the 24-block T5-XXL encode for the positive prompt — and the negative too whenever CFG is on (HD/Base default `true_cfg = 4.0`) — plus rebuilding RoPE tables and masks, per image. Flux hoists the encode outside its count loop.
- **Impact:** A count-8 CFG batch pays 16 redundant T5-XXL forwards (seconds each).
- **Suggested fix:** Split `denoise_with_schedule` into a prepare step (once) and a per-seed loop, mirroring flux's `run_denoise` shape.
- **Confidence:** High

#### [F-108] PuLID's identity stage runs with zero cancellation checks
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `mlx-gen-pulid/src/pulid_flux.rs:259-302`; secondary: `mlx-gen-flux/src/model.rs:300, 333-334`
- **Finding:** `generate_impl` runs SCRFD detection, BiSeNet parsing, ArcFace, the 24-block EVA-CLIP tower, the IDFormer (twice with real-CFG), builds up to 40 CA modules, then the backbone's T5-XXL + CLIP encodes — all before the first `req.cancel` consultation inside `run_flow_sampler`.
- **Impact:** Worst cancel latency exactly on the priciest pre-denoise stage — the F-018/F-019 class remediated elsewhere but not here.
- **Suggested fix:** Check `req.cancel.is_cancelled()` at the top of `generate_impl` and between the identity stages; same cheap pre-check at the top of `Flux1::generate`.
- **Confidence:** High

#### [F-109] `flux1_dev_control` silently ignores `req.use_pid` (and `LoadSpec::pid` / `ip_adapter`)
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `mlx-gen-flux/src/model_control.rs:122-167, 259-262`
- **Finding:** The base flux path errors loudly when `use_pid` is requested without a PiD overlay; chroma does the same; qwen's control variant wires PiD on its control path. `Flux1DevControl` neither loads `spec.pid` nor consults `req.use_pid` — the request validates and quietly decodes through the native VAE. `spec.ip_adapter` is likewise silently dropped at load.
- **Impact:** A knob that is a hard error on `flux1_dev` becomes a silent no-op on `flux1_dev_control` — the epic-7840 "no silent VAE fallback" convention violated on exactly one flux entry point.
- **Suggested fix:** Wire the flux PiD student into the control decode (mirror qwen), or reject `req.use_pid` in validate and error on `spec.pid`/`spec.ip_adapter` at load.
- **Confidence:** High

### Anima

#### [F-116] Add the gen-core-testkit conformance test anima's Cargo.toml already claims
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `mlx-gen-anima/Cargo.toml:26-29`
- **Finding:** `sceneworks-gen-core-testkit` is declared as a dev-dependency with a comment claiming capability-honesty/progress/cancel/seed/registry conformance coverage, but no test file references the testkit. Siblings have `tests/conformance.rs` (krea, z-image, +trainer twins).
- **Impact:** The conformance suite is exactly the harness that would catch F-115 (the ignored scheduler), cancel/progress gaps, and seed nondeterminism. The dep is dead weight and the comment misleads — "fixes don't travel to new crates" in its purest form.
- **Suggested fix:** Add `tests/conformance.rs` (and a trainer twin) mirroring krea/z-image, or remove the dev-dependency and its claim.
- **Confidence:** High

#### [F-117] Thread the training cancel flag into anima's preview-sample denoise
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `mlx-gen-anima/src/pipeline.rs:487-513` (`render_latent_with_enc`), used from `mlx-gen-anima/src/training.rs:703-760`
- **Finding:** In-training preview rendering builds a throwaway `CancelFlag::default()`, so cancellation is only observed between preview prompts. One preview = up to `sample_steps` × 2 DiT forwards plus a VAE decode, all uncancellable. Inference has the same shape at smaller scale (VAE decode after the sampler loop, no check). Inherited verbatim from z-image's `render_sample`.
- **Impact:** Cancel latency during training previews is a full multi-step 2B-DiT denoise + decode — the 07-01 systemic cancel theme inherited by new code.
- **Suggested fix:** Pass `req.cancel` into the preview render helpers; treat `Error::Canceled` from a preview as loop exit. Fix the z-image original too.
- **Confidence:** High

#### [F-118] Skip anima's uncond forward when guidance == 1.0 on CFG variants
- **Category:** efficiency
- **Severity:** Medium
- **Location:** `mlx-gen-anima/src/pipeline.rs:195-199, 239-245`
- **Finding:** `denoise` decides CFG purely on `variant.uses_cfg()`: base/aesthetic always encode the negative prompt and run two DiT forwards per step even at guidance 1.0, where the combine collapses to `v_c` exactly. The crate's own preview path already implements the skip.
- **Impact:** A `guidance: Some(1.0)` request on `anima_base` costs 2× the DiT compute for a bit-identical result.
- **Suggested fix:** Compute `uncond` only when `uses_cfg() && guidance != 1.0`, mirroring `render_preview_latent`.
- **Confidence:** High

#### [F-119] Extend anima's resume surface guard to factor shapes / rank–alpha
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `mlx-gen-anima/src/training.rs:260-283, 591-606` (with `src/train/checkpoint.rs:87-121`)
- **Finding:** `assert_resume_surface_matches` compares factor **key sets** only; keys don't encode rank. A run resumed with a different `cfg.rank`/`alpha` than the checkpoint passes the guard and trains at the checkpoint's rank, but `save_anima_lora` bakes `alpha/rank` from the **new** config — the final adapter is saved at a silently wrong scale (e.g. 2× attenuated).
- **Impact:** A rank/alpha change across a resumed run produces a valid-looking but mis-scaled adapter with no error — "structurally green, numerically wrong". Likely shared with z-image; fix in the shared checkpoint engine if so.
- **Suggested fix:** Compare factor shapes against the fresh-built `expected` params, or persist rank/alpha in resume metadata and reject a mismatch like the optimizer check does.
- **Confidence:** Medium

### Bernini

#### [F-133] Propagate the F-097 source-embedding hoist to `vit_one_step` (the registered full-`bernini` hot path)
- **Category:** efficiency
- **Severity:** Medium
- **Location:** `mlx-gen-bernini/src/forward.rs:458-548`
- **Finding:** The epic-9084 remediation added `embed_sources`/`velocity_pre` so identical conditioning sources are embedded once per step — but only wired into `guided_velocity` (the `bernini_renderer` id). `vit_one_step`, the per-step dispatch for the registered full `bernini` pipeline, still re-runs `embed_sources` per prediction: 3× per step in `VaeTxtVitWapg` (primary t2i/edit), up to 4× in `Rv2vWapg`, across ~40 steps.
- **Impact:** Redundant patch-embed convs + RoPE builds + phase folds several times per step for the whole denoise; the fix is bit-identical by construction.
- **Suggested fix:** Build the embedded segments once in `vit_one_step` and route through `velocity_pre`; optionally hoist above the step loop since sources never change within a run.
- **Confidence:** High

### SDXL family

#### [F-141] Guard the PEFT LoKr metadata `rank`/`alpha` against zero before deriving the scale
- **Category:** security
- **Severity:** Medium
- **Location:** `mlx-gen-sdxl/src/adapters.rs:511-518, 549-581`
- **Finding:** `merge_one_lokr` parses `rank`/`alpha` from file metadata; an explicit `rank = "0"` parses fine, `alpha` defaults to `rank`, and the scale becomes `0/0 = NaN` — baked into the packed factors or merged as a NaN delta via `merge_dense_delta`. The two sibling paths in the same file are guarded (`lora_delta` rejects rank 0 per sc-5252/F-002; third-party LoKr/LoHa hardened per F-010); this middle path was missed. Reachable from SDXL, Kolors, and InstantID.
- **Impact:** A corrupt/adversarial LoKr file NaN-poisons every subsequent render while the load reports success.
- **Suggested fix:** Validate `rank > 0.0` and finite `alpha` after the metadata parse, or add the guard once inside `reconstruct_lokr_delta`/`build_lokr_factors` so every consumer inherits it.
- **Confidence:** High

### PiD (near-universal decoder overlay — defects multiply across ~13 consumers)

#### [F-149] Propagate the PiD budget guard + watchdog tiling to InstantID's direct decoder mint
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `mlx-gen-instantid/src/model.rs:284-300` (vs the fixed seam at `mlx-gen-pid/src/engine.rs:188-227`)
- **Finding:** The F-013 remediation (budget `guard` → `plan_tile_edge` → `with_tiling`) lives inside `resolve_pid_decoder_at_sigma`. Every registered consumer routes through it — except InstantID, whose struct-API `pid_decoder_for` calls `engine.decoder(...)` directly: no budget guard, no tile plan, always the whole-image forward. InstantID validates only multiple-of-8 dims, so a 1536² request super-resolves at 6144² — the geometry `budget.rs` documents as tripping the Metal IOGPU watchdog (~100 s abort) or OOM.
- **Impact:** A plausible InstantID+PiD request dies in an opaque Metal/OOM failure instead of tiling or refusing — the T1 "sibling consumer misses the seam fix" chain.
- **Suggested fix:** Mirror the resolve seam in `pid_decoder_for`, or expose a `mint_planned_decoder(...)` helper in `mlx-gen-pid` so the policy has exactly one instance.
- **Confidence:** High

#### [F-150] Stop multiplying the PiD budget/tile plan by `req.count` — every consumer decodes B=1
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `mlx-gen-pid/src/engine.rs:196-218`; `mlx-gen-pid/src/budget.rs:169-191, 289-297`
- **Finding:** `guard` prices the resident floor as `count` full-resolution buffer sets and `plan_tile_edge` gets `b = req.count` — but the minted decoder is shared across the count loop and every consumer (verified: krea, ideogram, and the sampler's noise allocation) decodes one latent at a time. Concurrent peak never scales with `count`.
- **Impact:** False typed refusals and needlessly shrunken tiles for multi-image requests: count=8 at 2048² prices ~38 GiB against the ~5 GiB one decode actually holds.
- **Suggested fix:** Plan and guard with the per-decode batch (1, or the actual latent batch at decode time); update `guard_counts_the_batch` to assert the per-decode semantics.
- **Confidence:** High

#### [F-151] Hoist Ideogram's step-invariant `llm_cond_proj` out of the per-step forward
- **Category:** efficiency
- **Severity:** Medium
- **Location:** `mlx-gen-ideogram/src/transformer/model.rs:166-186`; fed at `mlx-gen-ideogram/src/pipeline.rs:586-594, 646-648`
- **Finding:** F-029's fix hoisted the role masks/MRoPE into `prepare`, but the largest step-invariant compute stayed per-step: `rms_norm(llm_features)` + the 53248×4608 projection over the full sequence + the indicator lookup depend only on `llm_features`/`prep`, yet run on every one of ~96 forwards per quality-mode image. The unconditional branch projects a `zeros` tensor and multiplies by an all-zero mask — a per-step ~2×10¹²-FLOP matmul whose result is provably zero.
- **Impact:** On the order of 10¹⁴ wasted FLOPs (tens of seconds) per 1024² quality image.
- **Suggested fix:** Compute the projected+masked LLM stream once in `prepare` and store it in `PreparedConditioning`; the per-step body reduces to `input_proj(x)·img_mask + prep.llm_plus_indicator`. Bit-identical.
- **Confidence:** High

#### [F-152] Replace PiD's arange `take_axis` gathers with strided splits on the hot attention path
- **Category:** efficiency
- **Severity:** Medium
- **Location:** `mlx-gen-pid/src/backbone/blocks.rs:57-58, 162-165`; `mlx-gen-pid/src/backbone/mod.rs:60-66`; `mlx-gen-pid/src/tiling.rs:135-138`; same class at `mlx-gen-ideogram/src/pipeline.rs:624, 644`
- **Finding:** The F-114/F-115 class fixed in qwen did not travel to the newer PiD crate: the joint `[txt, img]` output is split with host-built index vectors (up to 65 536 i32 at SR-4K) gathered in each of 14 patch blocks per step × 4 steps × tiles; `flash_sdpa` slices padded head_dim back with another gather per pixel-stream attention; `tiling::slice_axis` gathers full-res tensors per tile per step.
- **Impact:** Repeated host→device index uploads plus materializing gather kernels where zero-copy `split_sections` suffices, on the most expensive path in a crate wired into ~13 families.
- **Suggested fix:** `split_sections` at the fixed offsets in all four sites (and ideogram's `run_denoise`).
- **Confidence:** High

#### [F-153] Cache PiD's step-invariant host tables instead of rebuilding them per forward
- **Category:** efficiency
- **Severity:** Medium
- **Location:** `mlx-gen-pid/src/backbone/layers.rs:204-232, 269-284`; `mlx-gen-pid/src/backbone/mod.rs:154-163, 180-188`; `mlx-gen-pid/src/tiling.rs:181-182`
- **Finding:** Every `PidNet::forward` rebuilds with scalar host loops: the pixel positional table (`~268 MB` host alloc + 134M sin/cos per 2048² tile forward), both RoPE tables, and (tiled) the per-tile feather weights — all pure functions of grid geometry, invariant across the 4 sampler steps and identical for same-sized tiles. A 6144² decode recomputes the identical pos table 36 times (~9.6 GB of host churn). `pixel_pos_axis_scale` also re-reads its env var per call.
- **Impact:** Seconds of single-threaded CPU + H2D traffic per decode, serialized against the GPU, on the seam wired into ~13 families.
- **Suggested fix:** Memoize per `(h, w)` at `Sampler::run_inner` scope (at most a few distinct shapes per run); same for feather weights per fade pattern.
- **Confidence:** High

### SCAIL-2

#### [F-158] Add the self-validate call to scail2's `generate` path (only provider that skips it)
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `mlx-gen-scail2/src/pipeline.rs:125-131, 143-218`
- **Finding:** `impl_generator!`'s `generate` does not call `validate`, so every other provider re-validates at the top of its generate impl (z-image, krea, wan, bernini, svd, seedvr2 all verified). scail2's `run()` never does — a direct `Generator::generate` call bypasses the whole shared floor: size ceiling, count, sampler membership, conditioning allowlist, and the F-053 finiteness guard (`guidance: Some(NAN)` NaN-poisons a multi-minute render into garbage-as-success).
- **Impact:** One provider silently trusts the caller where 28 are defense-in-depth; the bug classes the shared validator exists to stop become reachable.
- **Suggested fix:** First line of `run()`: `self.descriptor.capabilities.validate_request(self.descriptor.id, req)?`.
- **Confidence:** High

#### [F-159] Stop advertising `MultiReference` — scail2 silently drops extra-character conditioning
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `mlx-gen-scail2/src/pipeline.rs:54-59, 196`
- **Finding:** The descriptor advertises `ConditioningKind::MultiReference`, so the floor accepts it — but `run()` never reads it (`additional` hardcoded to `Vec::new()`; the module doc says multi-reference "awaits the sc-5583 request contract"). A multi-reference job validates, renders, and returns success with the extra characters silently ignored.
- **Impact:** Silent input drop presented as success — the class the F-099 seedvr2 fix removed.
- **Suggested fix:** Remove `MultiReference` from the descriptor until sc-5583 lands (the floor will then reject it typed), or reject explicitly in `run()`.
- **Confidence:** High

### Sequential residency (epic 10834) — consolidated cross-crate findings

#### [F-173] Check `CancelFlag` at Sequential stage boundaries (8 sites)
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `mlx-gen-sdxl/src/model.rs:617-635`, `mlx-gen-z-image/src/model.rs:409-420`, `mlx-gen-qwen-image/src/model.rs:412-420` + `model_edit.rs:436-446` + `model_control.rs:448-458`, `mlx-gen-krea/src/model.rs:529-547` (+ `model_control.rs`), `mlx-gen-lens/src/registry.rs:344-355`
- **Finding:** The only inference-path cancel check is the per-step `step_gate` inside denoise. No wired crate checks `req.cancel` before Phase A (TE load + encode), between encode and `load_seq_heavy()`, or after the heavy load. Under Sequential the pre-first-step stretch now includes loading a ~15 GB text encoder from disk, a full encode, then loading a ~20 GB DiT+VAE — per generate. The `src/sampler.rs:181-182` doc claim "landing within ~1 model eval" is no longer true in Sequential mode; a pre-cancelled request runs the whole preamble.
- **Impact:** Cancellation latency of tens of seconds to minutes plus wasted disk I/O and a full peak-memory excursion on every cancelled Sequential job — on exactly the memory-constrained machines the feature targets.
- **Suggested fix:** `if req.cancel.is_cancelled() { return Err(Error::Canceled) }` at three points per Sequential generate (before encode, before `load_seq_heavy`, after it) — once, in the shared helper of F-175. Threading cancel into the per-layer component loaders (lens's per-layer prequantize eval is a natural checkpoint) is the fuller fix.
- **Confidence:** High

#### [F-174] Run the Sequential `clear_cache()` cleanup on error/cancel exits too (8 sites)
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `mlx-gen-sdxl/src/model.rs:1024-1029`, `mlx-gen-z-image/src/model.rs:343-359, 484-491`, `mlx-gen-qwen-image/src/model.rs:491-498` + `model_edit.rs:525-529` + `model_control.rs:515-518`, `mlx-gen-krea/src/model.rs:644-647` + `model_control.rs:279-283`, `mlx-gen-lens/src/registry.rs:283-300, 417-420`
- **Finding:** The end-of-generate tail (`drop(seq_heavy); clear_cache()`) and the post-encode TE cleanup run only on the success path. Any `?` exit — including the *routine* `Err(Canceled)` from `step_gate` — drops the components via unwind but skips `clear_cache()`, so MLX's buffer cache retains the multi-GB DiT/VAE working set (process RSS stays at the heavy-phase peak) until some later job clears it. Residency *state* stays consistent (the handle holds only the `LoadSpec`; the next generate works), and the `sequential_repeat_job_stays_bounded` tests cover only the success path.
- **Impact:** After a cancelled/failed Sequential job — the cancel/retry sequence a memory-constrained user is most likely to produce — the worker idles holding a DiT-sized cache, and a retry stacks the next TE load on top of it. The documented `max(TE, DiT+VAE)` bound silently doesn't hold on that path.
- **Suggested fix:** A small RAII guard whose `Drop` calls `mlx_rs::memory::clear_cache()` when armed for Sequential (or run the render body in an inner closure with cleanup on both arms). Fix once in the shared helper (F-175).
- **Confidence:** Medium — allocator reuse means the *active-memory* peak may still be bounded; the exposure is cache/RSS, which drives OS memory pressure but isn't measured by `get_peak_memory`.

#### [F-175] Hoist the eight-way copy-pasted residency scaffold into one shared seam
- **Category:** redundant
- **Severity:** Medium
- **Location:** `mlx-gen-sdxl/src/model.rs:197-208,452-495`; `mlx-gen-z-image/src/model.rs:110-119,319-385`; `mlx-gen-qwen-image/src/model.rs:92-100,314-380` + `model_edit.rs` + `model_control.rs`; `mlx-gen-krea/src/model.rs:181-189,377-463` + `model_control.rs:66-72,187-230`; `mlx-gen-lens/src/registry.rs:135-143,276-325`
- **Finding:** Eight near-verbatim copies of the same machinery: a `Residency`/`ControlResidency` enum, an `encode` doing load→encode→`eval`→drop→`clear_cache`, `load_seq_heavy`, a `heavy()` borrow-resolver with an identical `unreachable!`, and the `was_sequential` cleanup tail. Only the component types differ. F-172/F-173/F-174 are already the drift cost of this ring — any fix must be replicated 8 times, and the next family will paste a ninth.
- **Impact:** The workspace's #1 failure mode ("fixes don't travel") now has a fresh 8-instance ring at its center — created *after* epic 7778 (crate-boilerplate reduction) was opened for exactly this class.
- **Suggested fix:** A generic `Residency<Text, Heavy>` in `mlx-gen` core (or a macro beside `impl_generator!`) parameterized by two loader closures, providing `encode_with`, `load_seq_heavy`, `heavy`, the eval/drop/clear discipline, stage-boundary cancel checks, and an error-safe cleanup guard once. Fold into epic 7778.
- **Confidence:** High

#### [F-176] Advertise Sequential support so the advisory fallback is discoverable
- **Category:** bad-pattern
- **Severity:** Medium
- **Location:** `gen-core/src/generator.rs:492-514` (`Capabilities`), `gen-core/src/runtime.rs:56-71`
- **Finding:** `OffloadPolicy` is documented advisory, and `Capabilities` has no field advertising support. The ~18 provider crates that ignore it (incl. boogu — the largest available win at 20.6 GB DiT + 17.5 GB TE — and kolors, whose ChatGLM3-6B TE is the family's biggest ratio) are contract-conformant, but a fit-gate cannot distinguish "will bound peak to max(TE, DiT+VAE)" from "will OOM at the resident sum" without out-of-band knowledge. This is the enabling condition for F-172's silent OOM.
- **Impact:** Memory-planning callers must hardcode a per-model support list that will drift; a wrong entry fails as an OOM process kill rather than an error.
- **Suggested fix:** Add `supports_sequential_offload: bool` to `Capabilities` (default false), set it in the wired crates, and surface at least a log/warning (or opt-in strict error) when Sequential is requested of a non-supporting generator. Then wire boogu and kolors (both have the per-phase loaders in reach).
- **Confidence:** High

#### [F-177] Skip the per-generate PiD (+Gemma-2) load under Sequential when the request doesn't use PiD
- **Category:** efficiency
- **Severity:** Medium
- **Location:** `mlx-gen-sdxl/src/model.rs:425-431`, `mlx-gen-z-image/src/model.rs:273-279`, `mlx-gen-qwen-image/src/model.rs:241-246`, `mlx-gen-krea/src/model.rs:359-365`, `mlx-gen-lens/src/registry.rs:253-259`
- **Finding:** `load_heavy`/`load_seq_heavy` takes no request, so when the `LoadSpec` carries `pid`, every Sequential generate loads the PiD student **plus the Gemma-2-2b caption encoder** from disk — even for requests with `use_pid == false`, where it is never touched — and holds it through denoise although it participates only at decode. The documented peak `max(TE, DiT+VAE)` silently becomes `max(TE, DiT+VAE+PiD+Gemma)`.
- **Impact:** Seconds of wasted disk I/O and several GB of unnecessary heavy-phase footprint per Sequential request on PiD-equipped specs — enough to erase much of the headline memory win.
- **Suggested fix:** Thread `req.use_pid` into `load_seq_heavy` (Sequential only) and skip the PiD overlay when false; ideally defer the load to after the denoise loop.
- **Confidence:** High

---

## Low findings

### gen-core

#### [F-003] Floor the norm denominator in APG's norm-threshold clamp
- **Category:** bad-pattern | **Severity:** Low | **Location:** `gen-core/src/guidance.rs:211-217`
- **Finding:** `normalize_diff`'s threshold branch computes `num / ‖diff‖` without flooring the norm, contradicting the `GuidanceOps::div` contract. Zero-norm diff → `inf`/`NaN`; the CPU reference accidentally recovers (`f32::min(NaN, 1.0) = 1.0`) but MLX/candle `minimum` propagates NaN — backend-dependent behavior on the exact case the guard exists for.
- **Impact:** With cond == uncond (empty-negative CFG) and `norm_threshold > 0`, a tensor backend can NaN-poison guidance where the host reference stays finite — a cross-backend parity trap.
- **Suggested fix:** `clamp_min(norm, NORMALIZE_EPS)` before the divide.
- **Confidence:** High

#### [F-004] Add an upper bound for `steps` (and video counters) to the shared validation floor
- **Category:** security | **Severity:** Low | **Location:** `gen-core/src/generator.rs:563-572`
- **Finding:** The floor enforces `steps >= 1` but no ceiling; `steps: Some(u32::MAX)` (and `frames`/`fps`/`duration`) validate and start a multi-billion-step denoise.
- **Impact:** Request-reachable effective hang, recoverable only via cancel.
- **Suggested fix:** A `max_steps` capability or a generous fixed sanity cap (e.g. 1000); same for video counters.
- **Confidence:** Medium

#### [F-005] Guard `lcm_style_timesteps` against `original_steps > num_train_timesteps`
- **Category:** bad-pattern | **Severity:** Low | **Location:** `gen-core/src/sampling.rs:156-180`
- **Finding:** With `original_steps > num_train_timesteps`, `k = 0`, every origin entry is `-1_i64`, and `as usize` wraps to 2^64−1, later feeding an out-of-bounds slice panic in `LcmPolicy::denoised_coeffs`. Not reachable from today's constant call sites, but a `pub` policy function with a silent wrap the neighboring clamps imply is handled.
- **Impact:** Latent panic footgun on a public entry point.
- **Suggested fix:** Clamp `original_steps` to `..=num_train_timesteps` (or floor `k` at 1).
- **Confidence:** High

#### [F-006] Advertise control-training capability on `TrainerDescriptor` (sc-10163 gap)
- **Category:** bad-pattern | **Severity:** Low | **Location:** `gen-core/src/train.rs:95-99, 181-189, 275-286`
- **Finding:** The new control-training contract adds `control_type`/`control_image_path`, but `TrainerDescriptor` has no `supports_control` bit and no shared floor for "control_type set ⇒ every item carries a control image" — each trainer must re-implement the check, and the testkit can't exercise it (see F-055 for a trainer already silently ignoring the fields).
- **Impact:** The worker cannot introspect control-training support; per-family re-implementation is the fixes-don't-travel pattern.
- **Suggested fix:** Add `supports_control` (or `control_types`), a shared `validate_control_request` helper, and a testkit negative check.
- **Confidence:** Medium

#### [F-007] Reconcile the degenerate-schedule guards across the solver set
- **Category:** readability | **Severity:** Low | **Location:** `gen-core/src/sampling/solvers.rs:163-171, 284-333, 494-513`
- **Finding:** `Dpmpp2m`'s guard comment claims it "mirrors every sibling solver's leading guard", but `UniPc` has no terminal guard (leading-0 malformed schedule → `inf`) and `ErSde`'s history `expect(...)`s panic on an interior-0 schedule. Unreachable from the gen-core schedule builders, but the claimed invariant isn't held on `pub` entry points taking arbitrary `&[f32]`.
- **Impact:** Inconsistent robustness; a future provider-supplied custom schedule hits `inf` or a panic depending on solver choice.
- **Suggested fix:** Add the same leading guard to `UniPc`, replace the `ErSde` `expect`s, or document the descending-schedule precondition and fix the comment.
- **Confidence:** High

### Core `src/`

#### [F-011] Include dense-base LoHa/tucker materializations in the sc-10678 budget projection (and fix the false comment)
- **Category:** bad-pattern | **Severity:** Low | **Location:** `src/adapters/loader.rs:305-309, 336, 391-399`
- **Finding:** The pass-1 comment says dense groups "add nothing to `projected_materialize`", but `LycorisPlan::Dense` materializes the same `[out,in]` bf16 delta per target — merely excluded from the budget check. A full-coverage LoHa on a bf16 dense tier adds roughly the adapted-linear footprint again with no pre-flight.
- **Impact:** The OOM the guard prevents on packed tiers remains reachable on the dense tier; the comment misleads.
- **Suggested fix:** Count `projected_delta_bytes` for the `Dense` arm too (only refuses genuinely over-budget runs), or reword the comment as a deliberate exclusion.
- **Confidence:** High

#### [F-012] Surface (don't silently drop) a pass-2 re-resolution miss in `install_lycoris_groups`
- **Category:** bad-pattern | **Severity:** Low | **Location:** `src/adapters/loader.rs:369-401`
- **Finding:** Pass 2 skips a plan when `host.adaptable_mut` returns `None` — neither counted in `applied` nor recorded in `unmatched_paths`, contradicting the function's "surfaced, never silently dropped" contract. Unreachable today, but lazy/offloaded module trees are now a live pattern (epic 10834).
- **Impact:** A future non-deterministic host makes an adapter target vanish without trace.
- **Suggested fix:** Record the dotted path in `unmatched_paths` (or a typed internal error) in the `None` arm.
- **Confidence:** Medium

#### [F-013] Validate that fused BFL destinations share one `in_dim` in `install_bfl_lycoris`
- **Category:** bad-pattern | **Severity:** Low | **Location:** `src/adapters/loader.rs:1319-1343`
- **Finding:** The fused reconstruction shape uses the first destination's input dim and never checks the rest agree; a miswritten `bfl_targets()` entry surfaces as an opaque kron/reshape error far from the cause.
- **Impact:** A future BFL-table bug becomes a confusing downstream error instead of a named validation failure.
- **Suggested fix:** Error when `shape[1] != in_dim` for any subsequent destination, naming module and dims.
- **Confidence:** High

#### [F-014] Detect conflicting `.alpha` values in the BFL LoRA loader instead of nondeterministic last-wins
- **Category:** bad-pattern | **Severity:** Low | **Location:** `src/adapters/loader.rs:1188-1218`
- **Finding:** `apply_lora_bfl` iterates a HashMap and overwrites `parts.alpha` unconditionally, so a file with two alpha spellings for one target resolves by visit order. `apply_lora_peft` hardens the identical case with a hard "alpha conflict" error (`loader.rs:840-849`); this sibling didn't inherit it.
- **Impact:** A duplicated-key adapter installs at a nondeterministic scale across runs while reporting success.
- **Suggested fix:** Mirror the peft conflict check.
- **Confidence:** Medium

#### [F-015] Isolate loader test temp files per process
- **Category:** bad-pattern | **Severity:** Low | **Location:** `src/adapters/loader.rs:1758-1762` (contrast `src/weights.rs:213, 241`)
- **Finding:** Loader tests write fixed-name fixtures into a shared temp dir; `weights.rs` tests learned to suffix `std::process::id()`. Concurrent `cargo test` runs (parallel worktree sessions — the norm here) can interleave.
- **Impact:** Rare cross-process test flakes that look like real loader regressions.
- **Suggested fix:** Include the process id in `tmp()`'s directory.
- **Confidence:** High

### SD3 / SANA

#### [F-021] Encode the SD3 img2img reference latent once per request, not per image
- **Category:** efficiency | **Severity:** Low | **Location:** `mlx-gen-sd3/src/pipeline.rs:208-218`; `mlx-gen-sd3/src/model.rs:243-262`
- **Finding:** `denoise_img2img_cfg` runs `preprocess_init_image` + `vae.encode` inside the per-image loop; only the noise draw is seed-dependent.
- **Impact:** One redundant full VAE-encoder pass (f32, up to 1440²) per extra image.
- **Suggested fix:** Compute `clean` once in `generate_impl` and pass it in.
- **Confidence:** High

#### [F-022] Error on duplicate keys when merging SD3's T5 shards
- **Category:** bad-pattern | **Severity:** Low | **Location:** `mlx-gen-sd3/src/loader.rs:146-174`
- **Finding:** `load_t5_weights` merges shards with silent last-file-wins on duplicate keys; the SANA converter and `Weights::from_dir` both error on the same condition.
- **Impact:** A malformed snapshot loads wrong T5 weights and renders degraded output instead of failing diagnosably at load.
- **Suggested fix:** Error when `merged.insert(...)` returns `Some`.
- **Confidence:** High

#### [F-023] Don't silently fall back to the eos pad id for CLIP-bigG when `tokenizer_config.json` is unreadable
- **Category:** bad-pattern | **Severity:** Low | **Location:** `mlx-gen-sd3/src/loader.rs:63-91`
- **Finding:** `resolve_clip_pad_id` returns `CLIP_EOS_ID` on any missing/unparseable config. Correct for `tokenizer/` (L), but for `tokenizer_2/` (bigG) it silently reintroduces the sc-9581 bug (eos-padding corrupting the bigG penultimate hidden) on any partially-synced snapshot.
- **Impact:** Quality degradation on every sub-77-token prompt with no signal — the failure mode this seam exists to prevent.
- **Suggested fix:** Error (or log loudly) when the bigG config can't be resolved; keep the eos fallback for L only.
- **Confidence:** Medium

#### [F-024] Update the stale "Mask handling" section in the SANA text-encoder module doc
- **Category:** readability | **Severity:** Low | **Location:** `mlx-gen-sana/src/text_encoder.rs:26-32`
- **Finding:** The doc claims the 300-token attention mask "is not fed to the trunk"; the pipeline feeds it on every path and `CrossAttn::forward` applies it, documented "Required for correctness".
- **Impact:** A load-bearing rationale comment asserting the opposite of shipped behavior — a future reader could "simplify" the mask away citing it.
- **Suggested fix:** Rewrite to state the mask IS consumed and why.
- **Confidence:** High

#### [F-025] Drop SANA's duplicate `steps == Some(0)` check
- **Category:** redundant | **Severity:** Low | **Location:** `mlx-gen-sana/src/model.rs:311-313`
- **Finding:** The shared floor already rejects zero steps; the local re-check is unreachable duplication with a different error string.
- **Impact:** Two sources of truth for one rule.
- **Suggested fix:** Delete the local check.
- **Confidence:** High

### Lens

#### [F-031] Update the stale `models--SceneWorks--lens-turbo-mlx` cache fallback in the sc-9500 real-weight gates
- **Category:** bad-pattern | **Severity:** Low | **Location:** `mlx-gen-lens/src/text_encoder/gpt_oss.rs:1316-1329`
- **Finding:** The fallback still targets the retired `-mlx` repo-name suffix (sc-9517), so both sc-9500 `#[ignore]`d gates only run via the env override — the F-107 drift class recurring.
- **Impact:** The grouped-GEMM parity/perf gates silently stop resolving weights.
- **Suggested fix:** Point at the post-sc-9517 repo name (or try both), matching `trainer_e2e.rs`.
- **Confidence:** Medium

#### [F-032] Reuse the registry-resolved sigmas and cache lens's per-generation RoPE tables
- **Category:** efficiency | **Severity:** Low | **Location:** `mlx-gen-lens/src/registry.rs:361-366` vs `pipeline.rs:450`; `dit/transformer.rs:244-245, 326-327` + `dit/rope.rs:47-105`
- **Finding:** The registry resolves sigmas for the PiD plan, then `render` re-resolves the identical vector; `LensRope3d::forward` rebuilds the full host cos/sin tables on every DiT forward though the grid is constant across the schedule — the qwen F-114 RoPE-cache fix didn't travel (same as krea's F-079, since fixed there).
- **Impact:** Milliseconds per step of pure waste on the hot loop; a latent divergence seam between the two sigma resolutions.
- **Suggested fix:** Thread the resolved sigmas through; hoist the RoPE tables into a per-generation cache keyed on grid shape.
- **Confidence:** High

### FLUX.2

#### [F-036] Validate `image_guidance` beyond the kv variant — it silently no-ops elsewhere and accepts non-finite values
- **Category:** bad-pattern | **Severity:** Low | **Location:** `mlx-gen-flux2/src/model.rs:757-763, 908-918`; `model_control.rs:335-342`
- **Finding:** The F-087 rejection covers only the kv variant. On txt2img ids and `flux2_dev_control` a set `image_guidance` validates and is silently ignored; where honored, `+inf` passes the `> 1.0` filter and NaN-poisons the combine. The `FLUX2_IMG_GUIDANCE` env value is parsed without a finiteness check.
- **Impact:** Requests appear to succeed while a stated knob does nothing; a non-finite value yields a NaN render.
- **Suggested fix:** Reject `image_guidance` on variants that can't honor it; reject non-finite values where they can.
- **Confidence:** High

#### [F-037] Check cancellation before/between flux2's pre-denoise conditioning encodes
- **Category:** bad-pattern | **Severity:** Low | **Location:** `mlx-gen-flux2/src/model.rs:596-643`; `model_control.rs:223-299`
- **Finding:** The edit path VAE-encodes up to 8 references before the single pre-loop cancel check; the TE encode and img2img init encode run after it unchecked; `Flux2DevControl::generate_impl` has zero pre-loop cancel checks.
- **Impact:** A cancel at generate start is ignored for the whole conditioning stage (worst case ~8 VAE encodes at 2048²).
- **Suggested fix:** One check at the top of each `generate_impl` and one inside the per-reference encode loop.
- **Confidence:** High

#### [F-038] Make the flux2 edit-reference cap resolution-aware
- **Category:** bad-pattern | **Severity:** Low | **Location:** `mlx-gen-flux2/src/model.rs:66-75`; `config.rs:199-201`
- **Finding:** The F-027 cap counts references, not tokens; its "~4096 tokens each" rationale holds only at 1024². At the advertised `max_size = 2048`, 8 refs + target + txt ≈ 148k joint tokens — ~11× the configuration sc-6124 measured at ~104 GB unbounded. `eval_per_block` bounds the lazy-graph peak but not the per-block working set or O(n²) SDPA compute.
- **Impact:** The validate-time guard gives false assurance at high resolutions; the OOM class is still reachable inside the advertised surface.
- **Suggested fix:** Validate a joint-token budget computed from request dims × ref count.
- **Confidence:** Medium

#### [F-039] Deduplicate the img2img helpers shared by `Flux2` and `Flux2DevControl`
- **Category:** redundant | **Severity:** Low | **Location:** `mlx-gen-flux2/src/model.rs:480-521` vs `model_control.rs:146-191`
- **Finding:** `resolve_reference`, `encode_init_latents`, and the prompt-encode tuple builder are duplicated near-verbatim; the control copy already drifted cosmetically.
- **Impact:** Double maintenance on a numerically load-bearing conditioning chain (the cancel fix of F-037 must now land twice).
- **Suggested fix:** Hoist to free functions in `pipeline.rs`.
- **Confidence:** High

#### [F-040] Finish the F-111 slice-idiom remediation in flux2's per-step `run` closure
- **Category:** efficiency | **Severity:** Low | **Location:** `mlx-gen-flux2/src/model.rs:711-712`
- **Finding:** The generate loop still slices the leading target tokens with a host-built arange + `take_axis` per transformer forward per step; on the txt2img path the slice is a full-length no-op gather.
- **Impact:** Minor hot-loop overhead; inconsistent with the crate's post-F-111 convention.
- **Suggested fix:** `split_axis` when a reference is present; skip entirely when not.
- **Confidence:** High

#### [F-041] Update the stale flux2 `lib.rs` crate doc (S0-scaffold claim, klein-only framing)
- **Category:** readability | **Severity:** Low | **Location:** `mlx-gen-flux2/src/lib.rs:1-18`
- **Finding:** The crate doc still claims S0-scaffold status with guarded stubs; the crate ships five registered generators, the Mistral3 dev TE, a Pixtral vision tower, caption upsampling, KV-cache edit, and a converter.
- **Impact:** The first thing a reader sees misstates the crate's maturity.
- **Suggested fix:** Rewrite the header.
- **Confidence:** High

### SAM / face / depth

#### [F-044] Deduplicate `NUM_MASKMEM` between sam3's tracker and video modules
- **Category:** redundant | **Severity:** Low | **Location:** `mlx-gen-sam3/src/tracker.rs:714` and `mlx-gen-sam3/src/video.rs:39`
- **Finding:** The constant is declared privately in both modules; one is the temporal-pos wrap modulus (and the weight is `[7,…]`), the other sizes the bank windows. Nothing ties them together or to the checkpoint tensor.
- **Impact:** An edit to one copy silently mis-indexes the temporal-pos table or breaks the eviction guarantee.
- **Suggested fix:** Declare once `pub(crate)`; assert the tensor dim at load.
- **Confidence:** High

#### [F-045] Remove the duplicated `is_new` recomputation and the always-true `m > 0` guard in sam3 association
- **Category:** redundant | **Severity:** Low | **Location:** `mlx-gen-sam3/src/video.rs:495-520`
- **Finding:** The new-detection predicate is computed twice five lines apart; the `m > 0` conjunct is dead (early return above).
- **Impact:** A future edit to the rule must land twice; the dead conjunct suggests a guard that isn't one.
- **Suggested fix:** Compute `is_new` once; drop `m > 0`.
- **Confidence:** High

#### [F-046] Guard `Sam2VideoPredictor::init_state_from_pixels` against zero dims and an unvalidated clip tensor
- **Category:** bad-pattern | **Severity:** Low | **Location:** `mlx-gen-sam2/src/video_predictor.rs:531-544`
- **Finding:** The pixels-based constructor stores `video_h/w` and `images` unchecked: `video_w == 0` makes the prompt scale `inf` (silent garbage tracks); a wrong-shaped tensor errors opaquely later. The buffer-based sibling and the face/depth funnels all validate.
- **Impact:** Degenerate metadata yields silent garbage instead of a typed error.
- **Suggested fix:** Return `Result`, reject zero dims and non-`[T,3,1024,1024]` shapes.
- **Confidence:** High

#### [F-047] Deduplicate the depth crate's two hand-rolled host bilinear resamplers
- **Category:** redundant | **Severity:** Low | **Location:** `mlx-gen-depth/src/lib.rs:125-158` vs `mlx-gen-depth/src/preprocess.rs:30-54`
- **Finding:** Two near-identical half-pixel bilinear loops (u8→u8 vs u8→f32), both also re-implementing sam2's `bilinear_resize_f32` (itself the F-171 dedup) and gen-core's imageops.
- **Impact:** An edge-handling fix must land twice; the F-171 class recurs in a new crate.
- **Suggested fix:** One shared helper in the crate (or hoist a host bilinear into gen-core imageops).
- **Confidence:** High

### LTX

#### [F-052] Correct the grad-accumulation window after a mid-window resume
- **Category:** bad-pattern | **Severity:** Low | **Location:** `mlx-gen-ltx/src/training.rs:422-435` interacting with `:478-505`
- **Finding:** The F-017 window math assumes windows start at step 1; resume (sc-9560) can start mid-window, so with `save_every % accum != 0` the first post-resume flush divides a partial window by the full `accum`, under-scaling that update — beyond the documented dropped-partial-grads caveat, and nothing validates the divisibility.
- **Impact:** Silent training-dynamics skew on resumed runs; likely shared by the sibling trainers that got resume in the same commit.
- **Suggested fix:** Track the actual in-window count (reset on flush and resume), or validate/warn `save_every % accum == 0` when resuming.
- **Confidence:** High (mechanics); Medium (practical frequency)

#### [F-053] Validate `fps` in LTX — `fps: Some(0)` overflows in debug and degenerates silently in release
- **Category:** bad-pattern | **Severity:** Low | **Location:** `mlx-gen-ltx/src/positions.rs:107-130`; `mlx-gen-ltx/src/model.rs:463-469, 958-981`
- **Finding:** Neither the shared floor nor LTX checks `req.fps`. `fps = 0` → `duration = inf` → in `py_round`, `inf as i64` saturates to `i64::MAX` (odd) and `fi + 1` overflows — a panic in debug builds; in release, a 0-frame audio latent and `fps: 0` stamped on the output.
- **Impact:** Debug-build panic / silent zero-audio generation from one malformed field.
- **Suggested fix:** Reject `fps == Some(0)` in validate; make `py_round` saturate on non-finite input.
- **Confidence:** High

#### [F-054] Range-validate LTX conditioning strengths
- **Category:** bad-pattern | **Severity:** Low | **Location:** `mlx-gen-ltx/src/model.rs:657-786`; `conditioning.rs:139, 330-333`
- **Finding:** No Reference/Keyframe/VideoClip strength is clamped or validated on the request path (only `apply_replacement_mask` clamps a local copy while the same value passes unclamped as the clip strength). `strength > 1` → negative denoise mask → negative per-token σ timesteps and extrapolating blends.
- **Impact:** Silent garbage output from out-of-range strengths; every stage "succeeds".
- **Suggested fix:** Reject (or clamp, matching the replacement-mask precedent) strengths outside `[0, 1]`.
- **Confidence:** High

#### [F-055] Reject (not silently ignore) `control_type` / `control_image_path` in the LTX trainer
- **Category:** bad-pattern | **Severity:** Low | **Location:** `mlx-gen-ltx/src/training.rs:198-222`
- **Finding:** sc-10163 added the control-training contract fields; the LTX trainer (no control branch) accepts a request carrying them, trains a plain LoRA, and reports success. Its validator already rejects LoKr and unsupported optimizers with clear messages.
- **Impact:** A caller requesting control-branch training gets a valid-looking non-control adapter with no error. Likely applies to the other LoRA-only trainers (see F-006 for the shared-floor fix).
- **Suggested fix:** Error when `control_type.is_some()` or any item carries a control image.
- **Confidence:** Medium

### Wan

#### [F-060] Add the per-target packed-base check to wan's LoHa additive installer
- **Category:** bad-pattern | **Severity:** Low | **Location:** `mlx-gen-wan/src/adapters.rs:930-979`
- **Finding:** Both LoKr additive installers refuse to materialize a full delta on a packed base per target; the LoHa installer has no such check — safety rests entirely on the call-site file-level guard being invoked first.
- **Impact:** A future caller reaching the installer directly on a packed base silently materializes the ~28 GB/expert dense delta — the OOM the sc-10051 rejection exists to prevent.
- **Suggested fix:** Return the sc-10051 `Error::Unsupported` when `lin.is_quantized()`, mirroring the LoKr installers.
- **Confidence:** High

#### [F-061] Make wan dual-expert resume all-or-nothing
- **Category:** bad-pattern | **Severity:** Low | **Location:** `mlx-gen-wan/src/training.rs:585-609`
- **Finding:** The resume loop restores each expert independently: with one bundle missing/corrupt, `start_step` advances to the found expert's step while the other keeps fresh factors and a zeroed optimizer, silently skipping its first `~start_step/2` micro-steps against a shifted replay.
- **Impact:** A partially-present checkpoint pair produces a silently half-trained MoE adapter pair.
- **Suggested fix:** Require both bundles with matching `meta.step`; error (or restart with an explicit warning) otherwise.
- **Confidence:** High

#### [F-062] Deduplicate wan's fold/additive LoRA grouping and two-pass spec loops
- **Category:** redundant | **Severity:** Low | **Location:** `mlx-gen-wan/src/adapters.rs:302-349` vs `:739-797`; `:526-542` vs `:669-685`
- **Finding:** The additive installer re-copies `merge_one`'s key-grouping loop nearly verbatim, and the alpha/rank resolution already drifted between copies (F-058 is the concrete cost).
- **Impact:** Format tweaks must land twice; the rank-0 divergence already happened.
- **Suggested fix:** Extract shared `group_lora_factors` and a two-pass driver.
- **Confidence:** High

### Krea

#### [F-074] Fix the `weight_p1` precedence contradiction between code and its doc/test
- **Category:** readability | **Severity:** Low | **Location:** `mlx-gen-krea/src/transformer/block.rs:41-61, 658-679`
- **Finding:** `RmsScale::from_weights` returns the pre-folded `*.weight_p1` whenever it exists, but the test doc states the pre-folded variant must win "only when the raw one is absent"; no both-present case is tested.
- **Impact:** A snapshot carrying both spellings silently resolves opposite to the documented contract while tests stay green.
- **Suggested fix:** Decide the precedence, fix the comment, add the both-present assertion.
- **Confidence:** High

#### [F-075] Update stale converter references in krea's control branch
- **Category:** readability | **Severity:** Low | **Location:** `mlx-gen-krea/src/control.rs:93-110`
- **Finding:** The doc claims a converter pre-unfolds the RMSNorm scales and the error message points at `examples/krea-control-convert.rs` — but the convert step was dropped (native `weight_p1` load, sc-8465) and that example no longer exists.
- **Impact:** A load-failure message sends the operator to a nonexistent tool; doc contradicts the module header.
- **Suggested fix:** Reword to the `weight_p1`-verbatim contract; point the error at the expected overlay artifact.
- **Confidence:** High

#### [F-076] Use the descriptor id in krea's img2img single-reference error
- **Category:** readability | **Severity:** Low | **Location:** `mlx-gen-krea/src/model.rs:680-688`
- **Finding:** `single_reference` errors with a hardcoded `"krea_2_turbo: …"` though it is reached on the `krea_2_raw` path too.
- **Impact:** Misleading diagnostics on Raw img2img.
- **Suggested fix:** Thread the id in, like `validate_request` does.
- **Confidence:** High

#### [F-077] Re-export the krea edit surface from the crate root and refresh the stale slice plan
- **Category:** readability | **Severity:** Low | **Location:** `mlx-gen-krea/src/lib.rs:27-36, 55`
- **Finding:** The crate-root re-exports omit the edit API (`load_edit`, `edit_descriptor`, `KREA_2_EDIT_ID`); the module doc's slice plan still ends before Raw/training/edit/control/residency all landed.
- **Impact:** Inconsistent public surface; onboarding doc misstates status.
- **Suggested fix:** Add the edit items; compress the slice plan to reality.
- **Confidence:** High

#### [F-078] Replace arange-gather sequence splices in krea's grounded encoder with contiguous splits
- **Category:** efficiency | **Severity:** Low | **Location:** `mlx-gen-krea/src/text_encoder/encoder.rs:123-148, 221-233`
- **Finding:** `slice_seq`/`replace_seq` build host arange vectors + `take_axis` per call — twice per deepstack layer per grounded encode — and `stack_and_trim`'s prefix drop does the same. The gather-vs-split pattern fixed in qwen (F-114) and in krea's own DiT, reintroduced in the new P2 encoder code.
- **Impact:** Minor per-encode overhead ×2 CFG branches; a solved bug-class regressing into new code.
- **Suggested fix:** `split_axis` at the fixed offsets, as the DiT does.
- **Confidence:** High

### CI / supply chain / docs

#### [F-085] core-llm remains the workspace's only mutable git ref (deliberate, mitigated)
- **Category:** security | **Severity:** Low | **Location:** `Cargo.toml:51-65`
- **Finding:** `core-llm = { git = ..., branch = "main" }`. Rev-pinned in #639 per F-048, then reverted with a now-correct rationale: mlx-llm's own core-llm dep is `branch = "main"`, and differing source ids would split the lock into two core-llm packages and fork the `inventory` registry statics. `Cargo.lock` pins the commit and CI runs `--locked`.
- **Impact:** Residual exposure limited to a reviewed re-lock pulling upstream main.
- **Suggested fix:** Land the true fix upstream (move mlx-llm off `branch = "main"`), then rev-pin both; track as a story if not already.
- **Confidence:** High

#### [F-086] Add dependabot updates for the SHA-pinned GitHub Actions
- **Category:** security | **Severity:** Low | **Location:** `.github/` (no `dependabot.yml`)
- **Finding:** Actions are now SHA-pinned (F-046 fixed) but no update mechanism exists, so the pins only move by hand.
- **Impact:** Pins fossilize; security fixes in the cache action arrive only if someone remembers.
- **Suggested fix:** Minimal `dependabot.yml` with `package-ecosystem: github-actions`.
- **Confidence:** High

#### [F-087] Re-bless CHECKSUMS.txt — it covers a shrinking fraction of the golden set
- **Category:** bad-pattern | **Severity:** Low | **Location:** `tools/golden/CHECKSUMS.txt` (36 entries, last blessed 06-09; ~150 dump scripts now)
- **Finding:** The golden-shift tripwire is inert for every family added since early June (krea, lens, sd3, sana, bernini, wan, ltx, sensenova). The gap is at least documented (the F-153 caveat).
- **Impact:** "Unexpected golden shift after re-dump" goes undetected for the majority of the catalog.
- **Suggested fix:** Re-bless after the next full re-dump; have each dump script append its own entry.
- **Confidence:** High

#### [F-088] Deduplicate anima's tokenizer assets and reconsider the 13MB `include_str!`
- **Category:** efficiency | **Severity:** Low | **Location:** `mlx-gen-anima/src/tokenizer.rs:23-25`, `mlx-gen-anima/assets/` (11MB + 2.3MB)
- **Finding:** Vendored `include_str!` tokenizers are the established pattern, but this is 3× the previous largest (~13.3MB baked into every consumer's binary, runtime-parsed per load), and the T5 asset is noted as "shared with mlx-gen-chroma" yet committed as a byte-copy.
- **Impact:** ~13MB permanent binary weight per linked consumer; duplicated repo bytes.
- **Suggested fix:** Deduplicate the T5 asset against chroma's; consider snapshot-dir load with the vendored copy as fallback.
- **Confidence:** High on facts; Medium on whether change is warranted.

#### [F-089] Move (or bless) anima's golden-generator scripts — they break the tools/ convention
- **Category:** bad-pattern | **Severity:** Low | **Location:** `mlx-gen-anima/tests/fixtures/gen_anima_parity_goldens.py`, `gen_anima_stage7_golden.py`
- **Finding:** Every other family's dump scripts live in `tools/` as `dump_*.py`; anima's two live in `tests/fixtures/` as `gen_*.py`. The scripts themselves are exemplary (subsampled <200KB goldens).
- **Impact:** An agent following the documented convention won't find anima's regeneration path; two homes invite divergence.
- **Suggested fix:** Move to `tools/`, or explicitly bless colocation in CLAUDE.md — pick one.
- **Confidence:** High

### Z-Image / Boogu

#### [F-095] Cap boogu prompt length — the tokenizer never truncates
- **Category:** bad-pattern | **Severity:** Low | **Location:** `mlx-gen-boogu/src/tokenizer.rs:60-78`; consumed at `pipeline.rs:207, 292, 375, 470`
- **Finding:** `BooguTokenizer::encode` never truncates; the `max_length: 1280` config field is documented inert. z-image pads/truncates to 512; boogu's TE forward and DiT joint sequence scale with unbounded prompt length.
- **Impact:** A pathological prompt produces an arbitrarily long TE forward and inflates every denoise step's joint attention — a request-reachable resource sink, likely divergent from the reference processor.
- **Suggested fix:** Truncate to the reference max (1280 fits the config) or reject over-long prompts in validate.
- **Confidence:** Medium

#### [F-096] z-image accepts whitespace-only prompts — the boogu F-146 trim fix didn't travel back
- **Category:** bad-pattern | **Severity:** Low | **Location:** `mlx-gen-z-image/src/model.rs:516-521` vs `mlx-gen-boogu/src/model.rs:452-456`
- **Finding:** Boogu's validator was fixed to `trim().is_empty()`; z-image's shared validator (all four ids) still checks only `is_empty()`. A `"   "` prompt renders an effectively unconditioned image.
- **Impact:** The degenerate state accepted as a defect in boogu is silently accepted in the crate the empty-prompt class originated from.
- **Suggested fix:** `trim().is_empty()` + a whitespace test case.
- **Confidence:** High

#### [F-097] `resolve_reference` is triplicated across z-image, boogu, and qwen-image
- **Category:** redundant | **Severity:** Low | **Location:** `mlx-gen-z-image/src/pipeline.rs:463-479`, `mlx-gen-boogu/src/pipeline.rs:776-793`, `mlx-gen-qwen-image/src/model.rs:285`
- **Finding:** The single-Reference resolver (multi → error; strength fallback) is copied per crate; the img2img leaves it composes with already live in `mlx_gen::img2img`.
- **Impact:** Three copies of one boundary policy; the next semantic tweak will drift.
- **Suggested fix:** Hoist into `mlx_gen::img2img` next to `init_time_step`.
- **Confidence:** High

#### [F-098] Boogu Base img2img duplicates the whole Base generate body
- **Category:** redundant | **Severity:** Low | **Location:** `mlx-gen-boogu/src/pipeline.rs:279-350` vs `:195-264`
- **Finding:** `generate_base_img2img_with_progress` copies `generate_with_progress` end-to-end, differing only in initial latent and schedule slice; the turbo pair repeats the same clone-and-tweak. z-image solved the identical need with one shared `render_batch` + parameters.
- **Impact:** Four near-identical denoise bodies in one file; a CFG-combine or dtype fix must land several times (and F-093 shows the img2img copy already diverged behaviorally).
- **Suggested fix:** Fold img2img into the base entry via `start_step` + `clean: Option<&Array>` parameters.
- **Confidence:** High

#### [F-099] Trainer resume logs with a bare `eprintln!` from library code (workspace-wide)
- **Category:** bad-pattern | **Severity:** Low | **Location:** `mlx-gen-z-image/src/training.rs:475-489` (pattern stamped into all six family trainers by sc-9560)
- **Finding:** The resume path prints `"[F-125] resuming from step …"` to stderr instead of reporting through the `TrainingProgress` callback every other trainer event uses.
- **Impact:** An embedded engine writes uncontrolled stderr; a resume is exactly the event a UI wants surfaced.
- **Suggested fix:** Emit a progress variant for resume; drop the `eprintln!` across all six trainers.
- **Confidence:** High

### Qwen-Image

#### [F-102] Cap the qwen Edit MultiReference image count
- **Category:** bad-pattern | **Severity:** Low | **Location:** `mlx-gen-qwen-image/src/model_edit.rs:565-597`
- **Finding:** `validate_reference_images` guards dims/buffers/aspect per image but not **how many**; each reference is VAE-encoded (~600 packed tokens) and concatenated into the 60-block joint attention. The fork's multi-image path targets a small handful (sc-2529 validated 2-3).
- **Impact:** Dozens of references quadratically inflate joint attention — request-reachable OOM/watchdog abort instead of a clean validation error; each also lengthens the uncancellable encode stretch (F-173).
- **Suggested fix:** Reject `len() > N` (fork-validated max, e.g. 4-8).
- **Confidence:** Medium

### FLUX.1 / Chroma / PuLID

#### [F-110] PuLID silently drops `negative_prompt` when `true_cfg` is unset or ≤ 1
- **Category:** bad-pattern | **Severity:** Low | **Location:** `mlx-gen-pulid/src/pulid_flux.rs:277-303`
- **Finding:** The descriptor advertises `supports_negative_prompt: true` unconditionally, but the fake-CFG branch sets `flux_req.negative_prompt = None` — the request succeeds with the negative prompt having zero effect.
- **Impact:** A silently inert request knob — the false-capability class the crate itself fixed for stray conditioning.
- **Suggested fix:** Reject `negative_prompt` without `true_cfg > 1` in validate, or document + warn.
- **Confidence:** High

#### [F-111] Chroma's hot per-step sequence slices are still arange-gathers (F-111/07-01 fix didn't travel from flux)
- **Category:** efficiency | **Severity:** Low | **Location:** `mlx-gen-chroma/src/transformer.rs:104-107, 329-332, 744-751`
- **Finding:** Flux's txt/img splits moved to `split_axis`; chroma's `DoubleAttn::forward` and `forward_prepared` still materialize index vectors and gather contiguous ranges — ~21 gathers per forward, ×2 under CFG, per step; the per-block `rows()` modulation gathers are rebuilt per block per step.
- **Impact:** Avoidable per-step gather kernels + host index builds on the chroma hot path.
- **Suggested fix:** `split_axis` for the txt/img seam; hoist the `rows` index arrays per step.
- **Confidence:** High

#### [F-112] CLIP pooled selection and T5 head reshapes hardcode batch 1 despite the F-061 batch generalization in the same file
- **Category:** bad-pattern | **Severity:** Low | **Location:** `mlx-gen-flux/src/text_encoder.rs:168-177, 521-527`
- **Finding:** F-061 generalized `ClipAttention::forward` for B>1, but the pooled-token selection flattens `[s, 768]` with a single host argmax and `shape_t5`/`unshape_t5` hardcode batch 1 — the claimed B>1 capability doesn't exist end-to-end.
- **Impact:** Half-applied generalization; the code reads as batch-capable but shape-errors on B>1.
- **Suggested fix:** Finish it (per-row argmax, batched reshapes) or assert `B == 1` at entry and drop the claim.
- **Confidence:** High

#### [F-113] Dead public API: flux's `load_vae_from_source` has zero callers
- **Category:** dead-code | **Severity:** Low | **Location:** `mlx-gen-flux/src/loader.rs:107-129`
- **Finding:** No callers anywhere in the workspace — chroma/boogu/sd3 all use `load_vae(root)`.
- **Impact:** Unused public surface implying a single-file VAE load path nothing exercises.
- **Suggested fix:** Remove or fold back into `load_vae` unless a tracked story is about to consume it.
- **Confidence:** High

### Anima

#### [F-120] Cancelled resumed run that made no new progress returns Ok
- **Category:** bad-pattern | **Severity:** Low | **Location:** `mlx-gen-anima/src/training.rs:589-612, 766-768`
- **Finding:** `steps_run` initializes to `start_step`, so a resumed run cancelled before any new step skips the `steps_run == 0` Canceled guard, writes a final adapter identical to the checkpoint, and returns `Ok(… final_loss: 0.0)` — a loss never computed this run.
- **Impact:** The worker sees a completed job for a cancelled request with bogus telemetry.
- **Suggested fix:** Track new-session steps separately; return `Error::Canceled` when 0 and the flag is set.
- **Confidence:** Medium

#### [F-121] `WeightsSource::File` silently discards the given filename
- **Category:** bad-pattern | **Severity:** Low | **Location:** `mlx-gen-anima/src/loader.rs:80-111`
- **Finding:** For `File(dit)`, only the grandparent dir is kept and the canonical `diffusion_models/{variant.dit_filename()}` is re-joined — a renamed finetune is silently ignored in favor of the canonical name, or fails naming a file the caller never mentioned.
- **Impact:** Surprising behavior for the one source kind whose point is naming a specific file.
- **Suggested fix:** Load the exact path as the DiT, or error explicitly on a basename mismatch.
- **Confidence:** High

#### [F-122] Hoist anima's per-axis RoPE angle computation out of the per-position loop and cache per grid
- **Category:** efficiency | **Severity:** Low | **Location:** `mlx-gen-anima/src/rope.rs:74-100` (called from `transformer.rs:425, 544` per forward)
- **Finding:** Inside the t×h×w loop, three per-axis angle vectors are re-derived per position (~27k allocations, ~80× redundant trig at 1536²), and the whole table is rebuilt host-side on every DiT forward (60×/generation, 1×/training step) though it is a pure function of the grid.
- **Impact:** Host-side waste compounding in the training loop; trivially avoidable.
- **Suggested fix:** Precompute the three axis tables; memoize `CosmosRope` per latent grid.
- **Confidence:** High

#### [F-123] Replace anima's `head_cols` gather with a slice
- **Category:** efficiency | **Severity:** Low | **Location:** `mlx-gen-anima/src/transformer.rs:41-44, 130`
- **Finding:** `head_cols` materializes a 4096-element index array and `take_axis` to express `x[:, :2H]`, once per DiT forward in `AdaLayerNorm`.
- **Impact:** Minor per-forward overhead; indirect expression of a prefix slice.
- **Suggested fix:** Contiguous slice/split.
- **Confidence:** Medium

#### [F-124] Update the stale "prompt weighting not yet implemented" claim in anima's parity-golden docs
- **Category:** readability | **Severity:** Low | **Location:** `mlx-gen-anima/tests/parity_goldens.rs:14`
- **Finding:** The header trap-list says prompt weighting is unimplemented, but sc-10566 landed: the same file tests the implemented parsing and `src/prompt_weight.rs` exists.
- **Impact:** A reader gets the opposite of the truth about a shipped capability.
- **Suggested fix:** Update the bullet.
- **Confidence:** High

#### [F-125] Consolidate the five copies of anima's HF-snapshot glob helper
- **Category:** redundant | **Severity:** Low | **Location:** `mlx-gen-anima/tests/common/mod.rs:37-48` + four other copies (`tests/real_weights.rs`, `tests/parity_real_weights.rs`, `src/training.rs:1906-1917`, `tests/training.rs`/`tests/packed_adapters.rs`)
- **Finding:** The identical snapshot glob is hand-copied into at least five binaries; `tests/common/mod.rs` exists precisely to share it but only two include it.
- **Impact:** A cache-layout or repo-name change must be fixed in five places (see F-031 for how that class rots).
- **Suggested fix:** Include the common module from the other binaries.
- **Confidence:** High

#### [F-126] Simplify the self-subsuming assertion in anima's sigma-schedule test
- **Category:** readability | **Severity:** Low | **Location:** `mlx-gen-anima/src/pipeline.rs:526-531`
- **Finding:** `(x < 1e-5) && (x.abs() < 1e-5)` — the first clause is implied by the second and reads like an incomplete edit.
- **Impact:** Noise; the effective assertion is correct.
- **Suggested fix:** Keep the `.abs()` clause.
- **Confidence:** High

#### [F-127] `encode_t5_weighted` drops the EOS at the 512-token boundary
- **Category:** bad-pattern | **Severity:** Low | **Location:** `mlx-gen-anima/src/tokenizer.rs:132-136`
- **Finding:** The weighted path appends EOS only when `ids.len() < MAX_LEN`; a prompt filling all 512 slots ships without EOS, diverging from `encode_t5`'s truncation policy and breaking the documented "strict no-op" contract at that edge.
- **Impact:** Conditioning differs between maximal weighted and unweighted prompts; pathological prompts only.
- **Suggested fix:** Truncate to `MAX_LEN − 1` then always append EOS (matching `encode_t5`).
- **Confidence:** Medium

### SenseNova / Bernini

#### [F-134] Finish the F-144 logits-dtype remediation — three readback sites still rely on accidental f32
- **Category:** bad-pattern | **Severity:** Low | **Location:** `mlx-gen-sensenova/src/t2i.rs:794, 837, 1136-1139` (consumer `:965`)
- **Finding:** F-144 was fixed in `T2iModel::prefill` and `decode_logits` with explicit f32 casts, but `prefill_prefix` and `append_generated_image` return raw un-cast lm_head logits read as f32 downstream — safe only via accidental f32 promotion.
- **Impact:** A future bf16-purity change turns these into buffer misreads while the fixed sites keep working.
- **Suggested fix:** The same one-line `.as_dtype(Float32)` at both sources.
- **Confidence:** High

#### [F-135] Add cancel checks between Bernini's heavy non-loop stages
- **Category:** bad-pattern | **Severity:** Low | **Location:** `mlx-gen-bernini/src/bernini.rs:650-878`; `pipeline.rs:387-450`
- **Finding:** sc-9093 threaded cancel into the MAR loop, both denoise loops, and VAE decode, but the stages between check nothing: planner load (~15 GB), per-source ViT/VAE encodes, T5 encode, two sequential ~28 GB expert loads (+ optional quantize).
- **Impact:** Worst-case cancel latency on the full `bernini` id is a minute-plus while holding tens of GB.
- **Suggested fix:** Cheap checks at stage boundaries in both `generate_impl`s.
- **Confidence:** High

#### [F-136] Aggregate sensenova interleave progress instead of restarting per image
- **Category:** bad-pattern | **Severity:** Low | **Location:** `mlx-gen-sensenova/src/t2i.rs:1288-1297`
- **Finding:** Each generated image's denoise reports `1..=steps` afresh and the AR text phases emit nothing — the restarting-bar shape F-096(b) flagged for scail2, which bernini's F-038 fix solved with a folded 1-based bar.
- **Impact:** No monotone completion signal for the worker-consumed interleave path.
- **Suggested fix:** Fold: `total = max_images × steps`, offset `current` by images so far.
- **Confidence:** High

#### [F-137] Validate `fps >= 1` in bernini before `smart_video_nframes`
- **Category:** bad-pattern | **Severity:** Low | **Location:** `mlx-gen-bernini/src/bernini.rs:336-363, 687-692`; `vit_preprocess.rs:80-95`
- **Finding:** `fps: Some(0)` → `raw = inf` → `floor() as i64` saturates and the subsequent multiply overflows — debug-build panic, wrapped-then-clamped release value, and `fps: 0` stamped on the output (same class as LTX F-053).
- **Impact:** Request-reachable overflow panic in debug; silently wrong release behavior.
- **Suggested fix:** Reject `fps == Some(0)` in `validate_impl` (both ids).
- **Confidence:** Medium

#### [F-138] Remove or gate bernini's uncallable planner quantize methods
- **Category:** dead-code | **Severity:** Low | **Location:** `mlx-gen-bernini/src/vision.rs:354-361`, `connector.rs:84-95`, `clip_diff.rs:402-405`
- **Finding:** Three `pub` quantize methods have no callers, and the sc-5146 policy comment explains why they must not be called (group-64-misaligned vision linears; dense-required connector/clip-diff) — yet their doc comments advertise "(sc-5146 load-time quantization)".
- **Impact:** Dead API surface whose docs actively invite the misuse the policy forbids; `VisionTower::quantize` would error or corrupt at runtime.
- **Suggested fix:** Delete (additive to restore), or replace the docs with the policy note.
- **Confidence:** High

### SDXL family

#### [F-142] Make the Kolors ChatGLM3-6B prompt encodes cancellable
- **Category:** bad-pattern | **Severity:** Low | **Location:** `mlx-gen-kolors/src/registry.rs:263-330`
- **Finding:** Up to two full ChatGLM3-6B forwards (pos + neg, 256 tokens) plus the hoisted VAE init encodes run before the first cancel check. The F-019 fix (lens) covered the same class; ChatGLM3 is the workspace's second-largest TE and got neither the per-layer nor the between-encodes check.
- **Impact:** Cancel ignored for several seconds of 6B compute per request.
- **Suggested fix:** Checks at entry, between pos/neg encodes, after the VAE encodes (or a per-layer hook mirroring lens — with the F-029 eval caveat).
- **Confidence:** High

#### [F-143] Carry the F-082 packed-load guard into the Kolors IP-Adapter loader
- **Category:** bad-pattern | **Severity:** Low | **Location:** `mlx-gen-kolors/src/ip_adapter.rs:38-44`
- **Finding:** `load_kolors_ip_adapter` calls `cast_all(dtype)` unconditionally; the 07-01 F-082(b) named this exact site, and the remediation guarded all three SDXL sites but left this clone — a pre-quantized IP checkpoint's u32 codes/scales would be blanket-cast and corrupted.
- **Impact:** Latent (no packed Kolors IP tier ships today) but identical to the trap the sibling was fixed for, while epic 8506 expands packed coverage.
- **Suggested fix:** Replicate the `is_packed` guard, or hoist a shared guarded `cast_dense_all`.
- **Confidence:** High

#### [F-144] Reject a `spec.quantize` that mismatches a pre-quantized snapshot instead of silently no-opping
- **Category:** bad-pattern | **Severity:** Low | **Location:** `mlx-gen-sdxl/src/model.rs:340-344, 410-426`; `mlx-gen-kolors/src/model.rs:222-237` + `chatglm3.rs:185-204`; `mlx-gen-instantid/src/model.rs:308-317`
- **Finding:** All three crates' quantize seams no-op on an already-packed base: loading a packed Q8 snapshot with `spec.quantize = Some(Q4)` "succeeds" at the wrong bit-width with no diagnostic. (qwen/krea gained exactly this rejection as the F-076 fix — it didn't travel to the SDXL family.)
- **Impact:** A mis-pointed tier directory silently renders at a different quality/memory profile than requested.
- **Suggested fix:** Compare requested bits against packed-detected bits and error on mismatch.
- **Confidence:** High

#### [F-145] Deduplicate the mirrored from_ldm schedule reconstruction against the count-loop schedule build
- **Category:** redundant | **Severity:** Low | **Location:** `mlx-gen-sdxl/src/model.rs:712-764` vs `:789-813, 891-959`; `mlx-gen-kolors/src/registry.rs:421-482` vs `:498-641`
- **Finding:** The PiD `vp_capture_plan` resolution rebuilds the exact σ schedule the active mode will denoise as a ~40-line copy that must stay in lockstep by hand (each carries a "mirrors the build below" comment).
- **Impact:** A strength/slicing change that misses one side silently desyncs `keep`/`capture_sigma` from the actual trajectory — a wrong (not crashing) capture.
- **Suggested fix:** Hoist a per-mode `resolve_run_sigmas(req, …)` used by both.
- **Confidence:** High

#### [F-146] Align `req.strength` semantics in IP mode between SDXL and Kolors
- **Category:** bad-pattern | **Severity:** Low | **Location:** `mlx-gen-sdxl/src/model.rs:663, 1081-1097` vs `mlx-gen-kolors/src/registry.rs:323-339, 658-674`
- **Finding:** SDXL folds the request-level strength into the IP scale (`reference_strength.or(req.strength)`); Kolors keeps them separate (per-reference = IP scale, `req.strength` = init strength). The same worker payload yields materially different identity strength.
- **Impact:** Cross-crate contract drift that will surface as "same settings, different result" reports.
- **Suggested fix:** Pick one rule (Kolors' separation is cleaner) and document it on `Conditioning::Reference`.
- **Confidence:** Medium

### PiD group

#### [F-154] Make PiD's `maybe_capture` actually best-effort — it `unwrap()`s mid-decode
- **Category:** bad-pattern | **Severity:** Low | **Location:** `mlx-gen-pid/src/decoder.rs:136-169`
- **Finding:** The doc promises "a failure logs but never breaks the decode", yet two `as_dtype(...).unwrap()` calls panic on conversion failure while `PID_CAPTURE_LATENT` is set, inside production `decode`; `decode_tiled` never captures at all.
- **Impact:** Env-gated dev workflows only, but a panic there contradicts the stated contract.
- **Suggested fix:** Fold the casts into the existing log path; document or fix the tiled asymmetry.
- **Confidence:** High

#### [F-155] Deduplicate the seeded noise/ε draw in PiD's `sample` / `sample_tiled`
- **Category:** redundant | **Severity:** Low | **Location:** `mlx-gen-pid/src/sampler.rs:167-193 vs 200-231`
- **Finding:** The two production entries duplicate the key-split + normal-draw block verbatim; a comment even asserts they must stay byte-for-byte identical — the condition a shared helper would enforce structurally.
- **Impact:** A future RNG tweak applied to one entry forks the sequence between tiled and whole-image decodes, breaking the tiling A/B invariant.
- **Suggested fix:** Extract `draw_noise_eps(...)`.
- **Confidence:** High

#### [F-156] Deduplicate `CaptionEncoder::encode` / `encode_with_mask`
- **Category:** redundant | **Severity:** Low | **Location:** `mlx-gen-pid/src/caption.rs:116-152`
- **Finding:** `encode` is a strict subset of `encode_with_mask`, duplicated including the sel-vector construction.
- **Impact:** Drift risk in the select-index policy between PiD and the SANA consumer.
- **Suggested fix:** `encode` delegates to `encode_with_mask(...).0`.
- **Confidence:** High

#### [F-157] Bridge JoyCaption cancellation during prefill, not only on streamed tokens
- **Category:** bad-pattern | **Severity:** Low | **Location:** `mlx-gen-joycaption/src/model.rs:186-223`
- **Finding:** The gen-core cancel flag is mirrored onto the core-llm flag only inside the `StreamEvent::Token` handler; during the vision-encode + prompt prefill (the slowest phase) a cancel is never propagated.
- **Impact:** Cancellation latency equals the full SigLIP+Llama prefill; a cancel issued at submission is honored at token 1.
- **Suggested fix:** Mirror on every stream event, or mirror once before calling `generate`.
- **Confidence:** Medium

### SeedVR2 / SCAIL-2 / SVD

#### [F-160] Validate SVD's float micro-conditioning knobs for finiteness
- **Category:** bad-pattern | **Severity:** Low | **Location:** `mlx-gen-svd/src/model.rs:170-204, 369-374`
- **Finding:** `motion_bucket_id` and `noise_aug_strength` pass through unchecked; `noise_aug_strength = NaN` poisons the VAE image latent and every frame decodes to garbage-as-success.
- **Impact:** Silent garbage output from one malformed field — the F-053 class at a provider-specific knob.
- **Suggested fix:** Require both finite (and `noise_aug_strength >= 0`) in `validate_output_params`.
- **Confidence:** High

#### [F-161] seedvr2 video upscale silently ignores `req.count`
- **Category:** bad-pattern | **Severity:** Low | **Location:** `mlx-gen-seedvr2/src/registry.rs:175-197`
- **Finding:** `max_count: 8` is advertised and the image branch honors it, but the video branch returns exactly one video regardless.
- **Impact:** Silent request-field drop, asymmetric within one function.
- **Suggested fix:** Reject `count > 1` for video in validate, or honor with per-count seeds.
- **Confidence:** High

#### [F-162] seedvr2 image-path progress reports `Step{1,1}` per output instead of `{i+1, count}`
- **Category:** bad-pattern | **Severity:** Low | **Location:** `mlx-gen-seedvr2/src/registry.rs:200-223`
- **Finding:** An 8-image job surfaces eight consecutive "1/1" steps; the count axis is invisible (the restart shape F-096 fixed for scail2 segments). The trailing `Progress::Decoding` after `generate_video` returns is misleading but harmless.
- **Impact:** Progress UIs can't show multi-image jobs; appears stuck.
- **Suggested fix:** `Step { current: i + 1, total: req.count }`.
- **Confidence:** High

#### [F-163] seedvr2 cancel latency spans a whole chunk including the host color-correction loop
- **Category:** bad-pattern | **Severity:** Low | **Location:** `mlx-gen-seedvr2/src/pipeline.rs:509-534, 384-403`
- **Finding:** Cancel is checked only at chunk boundaries; the per-frame color correction is single-threaded host math (5-level wavelet + three full-frame sorts per frame — seconds at 1536²), so a 64-frame chunk adds a minute-plus of uncancellable host work.
- **Impact:** Worst-case cancel latency = one full chunk (GPU + host CC).
- **Suggested fix:** Pass `cancel` into `frames_from_decoded`, check per frame.
- **Confidence:** High

#### [F-164] seedvr2's spatially-tiled still-image path emits no per-tile progress
- **Category:** bad-pattern | **Severity:** Low | **Location:** `mlx-gen-seedvr2/src/pipeline.rs:280-304, 623-692`
- **Finding:** The tiled image path — the *normal* path for 2048²+ stills since the sc-8261 VAE cap — has no progress channel; the registry emits a single `Step{1,1}` before a run of a dozen full encode→DiT→decode tile passes. Cancel works; liveness doesn't.
- **Impact:** Minutes-long HD upscales look frozen — the symptom F-099 fixed for video.
- **Suggested fix:** Thread `on_progress` through and emit `Step{tile_idx, n_tiles}`.
- **Confidence:** High

#### [F-165] seedvr2 buffers every decoded chunk before assembly and color-corrects padding frames
- **Category:** efficiency | **Severity:** Low | **Location:** `mlx-gen-seedvr2/src/pipeline.rs:508-540`; `video.rs:135-155`
- **Finding:** All chunks' RGB8 frames accumulate before `assemble_overlap` (peak host memory ≈ 2× the whole video), though chunks only overlap their immediate predecessor; trailing padding frames are fully color-corrected then dropped.
- **Impact:** Avoidable ~2× host memory on long clips plus wasted seconds of host compute.
- **Suggested fix:** Stream assembly into the chunk loop; cap the CC count at `n − start`.
- **Confidence:** High

#### [F-166] scail2 preprocesses and pins the entire driving clip on device up front
- **Category:** efficiency | **Severity:** Low | **Location:** `mlx-gen-scail2/src/generate.rs:344-348, 148-171`
- **Finding:** All driving frames + masks are stacked into device-resident f32 arrays before any segment runs and stay pinned for the whole job (~2×1.8 GB for a 480-frame clip); only the current segment's 81-frame window is ever needed.
- **Impact:** Resident memory scaling with clip length, adjacent to the crate's OOM-critical decode phase.
- **Suggested fix:** Preprocess per segment from the host frames.
- **Confidence:** Medium

#### [F-167] scail2's sampler vocabulary is the legacy spelling, not the curated axis wan moved to
- **Category:** bad-pattern | **Severity:** Low | **Location:** `mlx-gen-scail2/src/pipeline.rs:66`
- **Finding:** Capabilities advertise `["unipc", "dpm++"]` while the wan family migrated to curated `uni_pc`/`euler`/`dpmpp_2m` with aliases (sc-7296). Curated spellings are rejected by scail2's validate even though `SolverKind::from_name` accepts them.
- **Impact:** Interop inconsistency across the wan family at the sampler-selection seam.
- **Suggested fix:** Advertise the curated names + aliases exactly as wan does.
- **Confidence:** Medium

#### [F-168] seedvr2 crate doc "Status" contradicts the shipped spatial-tiling behavior
- **Category:** readability | **Severity:** Low | **Location:** `mlx-gen-seedvr2/src/lib.rs:23-26`
- **Finding:** The Status section calls HD spatial tiling "a tracked follow-up" and says over-budget HD is refused — but sc-5201/sc-6067/sc-8261 shipped the tiling, and over-budget requests now tile.
- **Impact:** Misleads maintainers about actual memory behavior — the T7 doc-drift class.
- **Suggested fix:** Update the paragraph.
- **Confidence:** High

### Sequential residency (consolidated, continued)

#### [F-178] Update the stale "candle FLUX lane only" claims in the gen-core contract docs
- **Category:** readability | **Severity:** Low | **Location:** `gen-core/src/runtime.rs:62, 130, 189`
- **Finding:** Three doc comments still say only the candle FLUX provider honors Sequential; since sc-10839/11000/11006/11030/11101, ten MLX generators across five crates do.
- **Impact:** Contract readers conclude Sequential is a no-op on MLX — or over-trust it on the unwired variants (F-172).
- **Suggested fix:** Name the mechanism (or the F-176 capability flag) instead of a hardcoded provider list.
- **Confidence:** High

#### [F-179] Emit progress (or document silence) during Sequential in-generate component loads
- **Category:** bad-pattern | **Severity:** Low | **Location:** `gen-core/src/runtime.rs:329-334`; all eight generate bodies
- **Finding:** Progress monotonicity is preserved, but the multi-GB TE load, encode, and DiT+VAE load now happen inside `generate` with zero callbacks — up to minutes of silence in the window where a worker watchdog expects heartbeats. The contract has no `Loading` variant.
- **Impact:** False watchdog trips / dead spinners on Sequential jobs.
- **Suggested fix:** Add an additive `Progress::Loading`-style variant (audit consumers) and emit at phase boundaries, or document the extended pre-`Step` silence on `OffloadPolicy::Sequential`.
- **Confidence:** Medium

#### [F-180] Add a default-run (tiny-fixture) test of the Sequential state machine
- **Category:** bad-pattern | **Severity:** Low | **Location:** `mlx-gen-sdxl/tests/sequential_residency_real_weights.rs:91,137` + the four sibling files
- **Finding:** All Sequential coverage is `#[ignore]`d real-weight A/B tests (well-designed; never run in CI). Nothing in the default suite constructs a Sequential handle and drives `encode → load_seq_heavy → heavy` — including the request-reachable `unreachable!` guard and the adapter-reapplication path. (Note: anima has no residency implementation or test — it is not part of the wired set.)
- **Impact:** A refactor breaking the Sequential call-order invariant or dropping adapter re-application passes CI green.
- **Suggested fix:** Where a crate has committed tiny pipeline fixtures, one default-run Sequential-vs-Resident equality test.
- **Confidence:** High

#### [F-181] Sequential + `spec.quantize` re-quantizes the whole model on every generate
- **Category:** efficiency | **Severity:** Low | **Location:** `mlx-gen-z-image/src/model.rs:244-265`, `mlx-gen-qwen-image/src/model.rs:227-235` (+ edit/control), `mlx-gen-sdxl/src/model.rs:457-496`
- **Finding:** Under Sequential with a quantize-at-load spec on a dense snapshot, each generate re-loads dense bf16 and re-quantizes — repeated compute, and the dense transient means the per-phase peak is the *dense* component size, shrinking the memory win. Packed snapshots avoid both; nothing warns when the expensive combination is selected, and the module docs claim the `max(TE, DiT+VAE)` peak unconditionally.
- **Impact:** The slowest, least-bounded configuration is exactly the memory-constrained one likely to combine Q4/Q8 with Sequential; the A/B test would even pass.
- **Suggested fix:** Warn/document at load (Sequential + `spec.quantize` + dense snapshot); longer term, quantize once and cache the packed map in the Sequential variant.
- **Confidence:** Medium

---

## Informational

#### [F-008] Deduplicate the `lambda` helper between cfgpp and solvers
- **Category:** redundant | **Severity:** Info | **Location:** `gen-core/src/sampling/cfgpp.rs:52-57`
- **Finding:** `cfgpp::lambda` is a byte-identical private copy of `solvers::lambda`, including the numerically load-bearing `1e-12` floor. **Fix:** make the solver helper `pub(crate)`. **Confidence:** High

#### [F-009] Compute the beta-linspace fraction in f64
- **Category:** readability | **Severity:** Info | **Location:** `gen-core/src/sampling.rs:120-137`
- **Finding:** `scaled_linear` computes the linspace fraction in f32 then widens, while the comment claims f64-like-diffusers; ~1e-8 relative error, far below tolerances but an audit hazard when chasing sub-1e-4 schedule diffs. **Fix:** `i as f64 / (n-1) as f64`. **Confidence:** High

#### [F-016] Guard `copy_dir` against symlinked-directory recursion
- **Category:** security | **Severity:** Info | **Location:** `src/quant.rs:227-247`
- **Finding:** `path.is_dir()` follows symlinks: a symlinked directory in a source snapshot is traversed (cycle → stack overflow; foreign target → copied into a rehosted tier). Dev-only converter path. **Fix:** use non-following `entry.file_type()` and skip/bound directory symlinks. **Confidence:** Medium

#### [F-017] `TokenEmbedding::quantize` hardcodes group size 64 while the packed loaders are parametric
- **Category:** readability | **Severity:** Info | **Location:** `src/nn.rs:88-105`; same class at `mlx-gen-flux/src/text_encoder.rs:91-103` (literal `64` vs the file's own `GROUP_SIZE` const)
- **Finding:** The packed-load seams became group-size-parametric (gs-32 Boogu DiT), but the fresh-quantize side pins 64 — a future gs≠64 family can load but not produce a packed embedding. **Fix:** accept `group_size: Option<i32>` like `AdaptableLinear::quantize`; use `GROUP_SIZE` in the flux copy. **Confidence:** High

#### [F-026] Remove the unused deprecated `CLIP_PAD_ID` alias
- **Category:** dead-code | **Severity:** Info | **Location:** `mlx-gen-sd3/src/loader.rs:38-41`
- **Finding:** Zero references outside its definition, and it carries a documented footgun ("NOT the correct pad for CLIP-bigG"). **Fix:** delete or `#[deprecated]`. **Confidence:** High

#### [F-027] Unreachable duplicate-key branch in SANA's converter `load_map`
- **Category:** dead-code | **Severity:** Info | **Location:** `mlx-gen-sana/src/convert.rs:109-124`
- **Finding:** Iterates `Weights::keys()` (unique by construction) then checks for duplicates; the error branch can never fire. **Fix:** collect directly; move the rationale into a comment. **Confidence:** Medium

#### [F-028] DC-AE decode/encode are not cancellable (bounded window)
- **Category:** bad-pattern | **Severity:** Info | **Location:** `mlx-gen-sana/src/dc_ae.rs:449-464, 597-612`; `mlx-gen-sd3/src/pipeline.rs:236-239`
- **Finding:** One monolithic graph after the last per-step check; with SANA's envelope capped at 1024² (F-032 fix) the window is seconds. **Fix:** none required now; per-tile checks if the envelope rises. **Confidence:** High

#### [F-033] Drop `KvCache::append`'s unused return value and its extra clones
- **Category:** dead-code | **Severity:** Info | **Location:** `mlx-gen-lens/src/text_encoder/gpt_oss.rs:281-294`
- **Finding:** Returns `(k_all, v_all)` (cloned) that the sole caller discards. **Fix:** return `Result<()>`. **Confidence:** High

#### [F-034] Reword LensTrainer's single-use error messages
- **Category:** readability | **Severity:** Info | **Location:** `mlx-gen-lens/src/training.rs:143-153, 375-379, 432-433`
- **Finding:** A second `train()` fails with "text encoder already freed (caching after train loop)" — an internal invariant, not the caller-facing contract ("this trainer instance has already run"). **Fix:** reword; document single-use on the type. **Confidence:** High

#### [F-042] Validate `enhance_temperature` at the flux2 boundary
- **Category:** bad-pattern | **Severity:** Info | **Location:** `mlx-gen-flux2/src/model.rs:455-457`; `text_encoder/encoder.rs:311-317`
- **Finding:** A NaN temperature NaN-poisons the softmax draw and the garbage rewrite silently becomes the render prompt (not caught by the `Err`/empty fallback). **Fix:** reject non-finite in validate (F-053 pattern). **Confidence:** Medium

#### [F-048] Use (or delete) SAM3's dead text-config fields; bound caller-supplied token ids
- **Category:** dead-code | **Severity:** Info | **Location:** `mlx-gen-sam3/src/config.rs:194-201`; gather at `text.rs:169-173`
- **Finding:** `vocab_size`/`projection_dim` are never read, and `vocab_size` is exactly the missing cheap bounds check for the public raw-`input_ids` path (id ≥ 49408 → unchecked OOB gather). **Fix:** delete or wire as a bounds check. **Confidence:** High

#### [F-049] Promote `to_arcface_input`'s crop-size `debug_assert` to a typed error
- **Category:** bad-pattern | **Severity:** Info | **Location:** `mlx-gen-face/src/align.rs:285-293`
- **Finding:** The only public buffer funnel in the crate still on `debug_assert_eq!` after the F-020/F-081 typed-error wave. **Fix:** the same `Error::Msg` rejection as `warp_affine`. **Confidence:** Medium

#### [F-056] Replace `unwrap()` in the LTX vocoder debug helpers
- **Category:** bad-pattern | **Severity:** Info | **Location:** `mlx-gen-ltx/src/vocoder.rs:498-501, 547-548`
- **Finding:** The F-113 production guard landed in `forward`, but the `#[doc(hidden)]` diagnostics still unwrap `act_post` — a panic on HiFi-GAN configs. Not request-reachable. **Fix:** reuse the `ok_or_else` guard. **Confidence:** High

#### [F-057] Use a slice, not an index-array gather, for LTX's IC-LoRA token readback
- **Category:** efficiency | **Severity:** Info | **Location:** `mlx-gen-ltx/src/pipeline.rs:1081-1085`; `conditioning.rs:32-35`
- **Finding:** The F-114 gather-vs-split class, executed once per generation (negligible). **Fix:** `slice_axis` prefix. **Confidence:** High

#### [F-063] Classify wan adapter files once instead of re-reading them in the packed guard
- **Category:** efficiency | **Severity:** Info | **Location:** `mlx-gen-wan/src/adapters.rs:623-642, 690-707`
- **Finding:** Every adapter file is header-parsed twice (guard, then installer) — small cost, but a two-sources-of-truth seam. **Fix:** have the guard return the classifications. **Confidence:** Medium

#### [F-064] `install_thirdparty_delta` resolves the same adaptable target twice
- **Category:** efficiency | **Severity:** Info | **Location:** `mlx-gen-wan/src/adapters.rs:956-979`
- **Finding:** Fetches `adaptable_mut` for the shape, then `push_at` re-resolves; the LoKr siblings resolve once. **Fix:** resolve once, push directly. **Confidence:** High

#### [F-065] `WanAdapterFamily` is `pub` but has no external consumer
- **Category:** dead-code | **Severity:** Info | **Location:** `mlx-gen-wan/src/adapters.rs:579-604`
- **Finding:** Exported and documented "for packed-snapshot routing", but only used internally for one comparison. **Fix:** `pub(crate)`, or wire into the F-063 fix. **Confidence:** High

#### [F-066] Align the wan 5B and A14B "matched no module" error texts
- **Category:** readability | **Severity:** Info | **Location:** `mlx-gen-wan/src/model.rs:166-177` vs `:640-656`
- **Finding:** Two functions that exist "so the message can't drift" drifted from each other (LoRA-only hint on the 14B though both accept LoKr/LoHa). **Fix:** one shared message builder. **Confidence:** High

#### [F-067] Complete the hoisted-constant `eval` list in `denoise_vace_moe`
- **Category:** readability | **Severity:** Info | **Location:** `mlx-gen-wan/src/vace.rs:953-960`
- **Finding:** The pre-loop eval omits the high-expert RoPE caches and uncond embeds, inconsistent with `denoise_vace` and the stated F-023 intent; numerically harmless. **Fix:** add the arrays or a deliberate-omission comment. **Confidence:** High

#### [F-068] `vace_prep`'s `expect("validated present")` relies on call-site convention
- **Category:** bad-pattern | **Severity:** Info | **Location:** `mlx-gen-wan/src/model_vace.rs:104`
- **Finding:** Unreachable via the registry today, but a third VACE variant that forgets to validate turns a missing `ControlClip` into a panic. **Fix:** `ok_or_else` with the validator's message. **Confidence:** High

#### [F-079] Apply the F-079 prepare hoist to krea's trainer preview render
- **Category:** efficiency | **Severity:** Info | **Location:** `mlx-gen-krea/src/training.rs:890-934`
- **Finding:** `render_sample` re-runs `prepare` (text fusion + host RoPE) every preview denoise step; the inference paths hoist it. **Fix:** build the preps once, call `forward_prepared`. **Confidence:** High

#### [F-080] Krea edit silently ignores `strength`; smoke examples hardcode personal paths
- **Category:** bad-pattern | **Severity:** Info | **Location:** `mlx-gen-krea/src/model.rs:572-577, 701-715`; `examples/krea_edit_smoke.rs:49-60`; `examples/krea_control_smoke.rs`
- **Finding:** A `Reference.strength`/`req.strength` on `krea_2_edit` is documented-dropped but never rejected (silent no-op knob); the examples default to `/Users/michael/...` paths (env-overridable). **Fix:** reject strength on the edit id; derive example defaults from the HF cache glob. **Confidence:** High

#### [F-090] Fix stale workspace-size comments
- **Category:** readability | **Severity:** Info | **Location:** `rust-toolchain.toml:1` ("17-crate workspace"; actual 33 packages), `Cargo.toml:5` ("31 member crates"; actual 32)
- **Finding:** Counts in comments always rot. **Fix:** drop the numerals. **Confidence:** High

#### [F-091] Minor nits in the two new dump scripts
- **Category:** readability | **Severity:** Info | **Location:** `tools/dump_dcae_encode_golden.py:34-37`; `tools/dump_sd3_empty_negative_e2e_golden.py:51-67`
- **Finding:** One bakes a specific snapshot hash into its default dir (glob `snapshots/*` like siblings); the other computes embeds it never saves; several older docstrings still carry personal paths despite `_paths.py`. **Fix:** opportunistic cleanup. **Confidence:** High

#### [F-092] Single-consumer per-crate dep pins (wan, sdxl) — acceptable, note only
- **Category:** redundant | **Severity:** Info | **Location:** `mlx-gen-wan/Cargo.toml:22-23` (`unicode-normalization`, `zip`), `mlx-gen-sdxl/Cargo.toml` (`regex`)
- **Finding:** The only deps outside `[workspace.dependencies]`, each single-consumer — the hoist rule is satisfied; all git deps in `Cargo.lock` resolve to expected sources; secrets scan clean. **Fix:** hoist only if a second consumer appears. **Confidence:** High

#### [F-100] Boogu Edit silently ignores a per-reference `strength`
- **Category:** bad-pattern | **Severity:** Info | **Location:** `mlx-gen-boogu/src/model.rs:422-444, 469-489`
- **Finding:** Accepted by validation on `boogu_image_edit`, discarded by the Edit path, while the same field on Base/Turbo changes behavior materially. **Fix:** reject a non-`None` strength on the Edit id (same class as F-080). **Confidence:** High

#### [F-103] qwen advertises `supports_true_cfg` but never consumes `req.true_cfg`
- **Category:** bad-pattern | **Severity:** Info | **Location:** `mlx-gen-qwen-image/src/model.rs:47-50`; `pipeline.rs:110-117`
- **Finding:** All three descriptors set it; the pipelines derive CFG only from `req.guidance`. A flux-style caller's `true_cfg` is silently ignored. **Fix:** honor `true_cfg.or(guidance)` or reject set-but-unused. **Confidence:** Medium

#### [F-104] qwen control-tier converter derives provenance from the checkpoint's parent directory
- **Category:** bad-pattern | **Severity:** Info | **Location:** `mlx-gen-qwen-image/src/convert.rs:125-131`
- **Finding:** A single-file overlay in a mixed directory inherits an unrelated config/license into a *published* tier. Dev-tool only, but provenance accuracy matters (the F-045 rationale). **Fix:** explicit provenance-root parameter or a sanity check on the parent. **Confidence:** Medium

#### [F-114] `flux1_dev_control` duplicates the hyper profile defaults inline
- **Category:** redundant | **Severity:** Info | **Location:** `mlx-gen-flux/src/model_control.rs:212-216` vs `model.rs:477-482`
- **Finding:** The `(8, DEFAULT_GUIDANCE)` selection is re-open-coded. **Fix:** call `profile_defaults` (`pub(crate)`). **Confidence:** High

#### [F-128] Anima VAE: malformed decoder-upsample key silently becomes `resnets.-1`
- **Category:** bad-pattern | **Severity:** Info | **Location:** `mlx-gen-anima/src/vae.rs:39-51`
- **Finding:** `parse().unwrap_or(-1)` produces a garbage key that surfaces later as a missing-weight error naming a key the checkpoint never contained. **Fix:** pass the key through unchanged (or error) when the index doesn't parse. **Confidence:** High

#### [F-129] Anima's f32 memory-projection branch is unreachable in production
- **Category:** dead-code | **Severity:** Info | **Location:** `mlx-gen-anima/src/training.rs:954-961`
- **Finding:** `bf16` is always true at the only call site; the f32 branch runs only in a unit test ("No f32 Anima base exists"). **Fix:** keep (documented symmetry) or drop; no action required. **Confidence:** High

#### [F-130] Anima preview adapter-install failure aborts the whole training run despite the "best-effort" contract
- **Category:** bad-pattern | **Severity:** Info | **Location:** `mlx-gen-anima/src/training.rs:705-710`
- **Finding:** Render failures log-and-continue, but the preceding `adapter.install_as(...)?` propagates, killing an hours-long run for a preview-only step (inherited from z-image). **Fix:** treat like a render error. **Confidence:** Medium

#### [F-131] Anima's per-step training-noise key collides across adjacent request seeds
- **Category:** bad-pattern | **Severity:** Info | **Location:** `mlx-gen-anima/src/training.rs:622-628`
- **Finding:** `(seed + step)·2 + 1` makes seed S/step k identical to seed S+1/step k−1 — adjacent-seed runs share most noise draws offset by one step. Determinism preserved. **Fix:** multiplicative mix like the sigma key. **Confidence:** High

#### [F-132] Awkward double-struct construction for anima's Euler render options
- **Category:** readability | **Severity:** Info | **Location:** `mlx-gen-anima/tests/real_weights.rs:391-401`
- **Finding:** `GenOptions { sampler: Some(..), ..GenOptions { …, sampler: None } }` builds a full inner literal immediately overridden. **Fix:** derive `Clone` and use struct-update from the earlier `opts`. **Confidence:** High

#### [F-139] Align `read_mrope_config` with the F-097 corrupt-sidecar policy
- **Category:** redundant | **Severity:** Info | **Location:** `mlx-gen-bernini/src/bernini.rs:140-161`
- **Finding:** Still uses the retired `.ok()…unwrap_or(Null)` swallow on `qwen2_5_vl_config.json`, shielded only by call order (strict parse two lines earlier); also re-reads the file. **Fix:** parse once, derive both configs. **Confidence:** High

#### [F-140] Harden sensenova `smart_resize` against degenerate dimensions like its bernini sibling
- **Category:** bad-pattern | **Severity:** Info | **Location:** `mlx-gen-sensenova/src/t2i.rs:1621-1645`
- **Finding:** `pub`, i32 math: 0-dim input → NaN→0-sized target; huge dim overflows. The bernini port of the same upstream function does i64 with banker's rounding. End-to-end damage contained by downstream guards. **Fix:** i64 math + non-positive rejection. **Confidence:** Medium

#### [F-147] Kolors rejects txt2img outright when an IP-Adapter is loaded, unlike SDXL
- **Category:** bad-pattern | **Severity:** Info | **Location:** `mlx-gen-kolors/src/registry.rs:279-285` vs `mlx-gen-sdxl/src/model.rs:658-663`
- **Finding:** Kolors errors on any request without a Reference once `spec.ip_adapter` is set; SDXL treats the same load as ip_mode-off and renders plain txt2img. **Fix:** pick one contract (SDXL's matches diffusers) and align or document. **Confidence:** High

#### [F-148] Wire (or explicitly track) Sequential offload for Kolors — the family member that benefits most
- **Category:** efficiency | **Severity:** Info | **Location:** `mlx-gen-kolors/src/registry.rs:139-205`
- **Finding:** Contract-legal (advisory), but Kolors carries the largest TE-to-UNet ratio in the SDXL family (ChatGLM3-6B) and its trainer already implements the free-after-cache pattern. See F-176 for the discoverability fix. **Fix:** port the SDXL `Residency` pattern; file under epic 10834. **Confidence:** High

#### [F-169] seedvr2's `decoder_stage_localize` asserts nothing numeric
- **Category:** dead-code | **Severity:** Info | **Location:** `mlx-gen-seedvr2/src/vae.rs:549-613`
- **Finding:** Asserts shape only and prints the cosine; its DiT counterpart asserts `cos > 0.999`. A diverged VAE decoder passes green (real gate lives in `tests/vae_parity.rs`). **Fix:** assert a cosine floor on the final stage, or mark print-only. **Confidence:** High

#### [F-170] scail2 applies `apply_clean_history` twice back-to-back every step
- **Category:** redundant | **Severity:** Info | **Location:** `mlx-gen-scail2/src/generate.rs:555-593`
- **Finding:** The post-step pin is re-applied at the top of the next iteration; idempotent so correct, but an extra take/concat pair per step and a puzzle for readers. **Fix:** keep one site with an upstream-order comment. **Confidence:** Medium

#### [F-171] scail2 diff-patch merge can half-apply a module when only its bias delta mismatches
- **Category:** bad-pattern | **Severity:** Info | **Location:** `mlx-gen-scail2/src/lora.rs:232-253`
- **Finding:** The "never half-apply" doc holds for weight-mismatch but not the converse: a merged weight followed by a shape-mismatched `.diff_b` leaves a half-applied module, surfaced only in the eprintln report. Malformed adapters only. **Fix:** validate both deltas before writing either. **Confidence:** High

---

## Themes and systemic observations

1. **Remediation works — and the epic-9084 wave proves it.** All 28 tracked 07-01 findings verified fixed with real guards and regression tests. The review→Shortcut-stories→remediation-PRs loop is functioning; the findings below are about what the loop doesn't yet catch.

2. **"Fixes don't travel" is still the dominant failure mode, now with a measurable half-life.** Roughly a third of this wave's findings are a guard that exists in one place missing from a sibling — sometimes in the same file (wan rank-0: F-058; sdxl LoKr rank: F-141; BFL alpha conflict: F-014), sometimes across crates (validation floor: F-018/F-101/F-105/F-158; gather-vs-split: F-078/F-111/F-152; packed-cast guard: F-143; tier-mismatch rejection: F-144; whitespace prompt: F-096). Two structural fixes would retire most of the class: (a) hoist guards into the seam everyone calls (the F-141 fix belongs in `reconstruct_lokr_delta`, the F-149 fix in a PiD mint helper, the F-097 resolver in `mlx_gen::img2img`), and (b) make `caps.validate_request` the mandatory first line of every provider validate — four crates still hand-roll below the floor.

3. **The sequential-residency rollout shipped the anti-pattern the workspace already named.** Epic 10834 created an 8-way copy-pasted scaffold (F-175) with systematic gaps that now each need 8 fixes: no stage-boundary cancel (F-173), no error-path cache flush (F-174), no per-request PiD skip (F-177), no default-run test (F-180) — plus a silent partial rollout inside z-image itself (F-172, High) that is undetectable because the contract has no capability bit (F-176). The generic `Residency<Text, Heavy>` helper should land before the next family is wired, not after.

4. **Cancellation keeps regressing at the seams the denoise loop doesn't cover — and one fix went false-green.** The lens MoE cancel fix stopped working when sc-9500 removed the host sync its premise relied on (F-029): under a lazy-execution backend, a cancel check without a forced `eval` checks nothing. New uncancellable stretches: Sequential load phases (F-173), LTX tiled decode/audio/conditioning (F-051), PuLID's identity stack (F-108), Kolors ChatGLM (F-142), anima training previews (F-117), seedvr2 in-chunk host color correction (F-163), bernini stage seams (F-135), flux2 reference encodes (F-037), joycaption prefill (F-157). A testkit conformance check that trips cancel mid-*encode* (not mid-denoise) and asserts return latency would catch the whole class mechanically.

5. **The NaN/finiteness floor lags the request surface.** F-053 fixed `guidance`/`true_cfg`, but every knob added since ships without the check: eta/momentum/norm-threshold (F-001), `image_guidance` (F-036), SVD micro-conditioning (F-160), `enhance_temperature` (F-042), LTX strengths (F-054) — and four providers bypass the floor entirely (theme 2). The floor needs a rule: every `Option<f32>` on `GenerateRequest` is finite-checked centrally, so new fields inherit the guard by construction.

6. **"Advertised but inert" is the new quiet contract violation.** Anima's schedulers (F-115, High), scail2's MultiReference (F-159), qwen's `true_cfg` (F-103), PuLID's unconditional `negative_prompt` (F-110), krea control's `use_pid` (F-070), flux-control's `use_pid`/`ip_adapter` (F-109), seedvr2 video `count` (F-161), boogu/krea edit `strength` (F-100/F-080), LTX trainer control fields (F-055). The capability surface is the contract; every advertised-or-accepted field must be honored or typed-rejected. A testkit sweep asserting "setting any advertised knob changes the output or errors" is feasible for the cheap knobs.

7. **Doc drift now has a measured half-life of about a week.** CLAUDE.md was fixed on 07-03 and re-drifted by 07-10 (F-081); CODEGRAPH.md confidently describes a different project (F-083); README lags five families (F-082); stale S0/status headers persist in flux2 and seedvr2 (F-041/F-168); gen-core's residency docs name the wrong provider set (F-178). Docs that enumerate (counts, crate lists, provider lists) rot fastest — prefer mechanism descriptions over lists, and consider a CI check that greps CLAUDE.md's crate list against `Cargo.toml` members.

8. **Progress reporting is the least-conformant contract.** >100% (F-050), frozen-below-total (F-030), restarting bars (F-136/F-162), silent tiled paths (F-164), silent Sequential loads (F-179). The σ-derived monotone `step_gate` is correct — providers that wrap or re-count it break it. A testkit property (monotone, reaches total, Decoding emitted exactly once) would pin the whole contract.

## Coverage notes

- **Reviewed:** all 33 workspace packages (root `src/`, `gen-core`, `gen-core-testkit`, 30 provider crates), `.github/workflows`, all Cargo manifests + lockfile git sources, `.cargo/config.toml`, `rust-toolchain.toml`, `tools/` (Python scripts + metallib refresh), root `tests/`, `docs/`, and the top-level docs. Reviewer agents read all non-test source in their scopes in full; test files were read in full for the heavy-churn crates (anima, krea, qwen-image, lens, wan) and inventoried/spot-read elsewhere.
- **Excluded:** `_vendor/` (read-only third-party reference checkouts, gitignored), `tools/golden/` payloads (gitignored, regenerable), `target/`, committed binary fixtures' numeric contents (conventions and sizes were reviewed, tensor values were not re-validated).
- **Method limits:** static review only — no builds, tests, or real-weight runs were executed; findings that depend on runtime behavior are marked Medium/Low confidence with what would confirm them. Cross-backend claims (candle) were checked only where gen-core documents them. The four ~50MB sensenova fixtures (F-084) were assessed by size/convention, not content.
- **Prior-review linkage:** the 28 tracked 2026-07-01 findings were re-verified against current source by a dedicated agent (all fixed; table in the executive summary). Unchanged known-open items from prior reviews that are already tracked in Shortcut are not re-reported here.

