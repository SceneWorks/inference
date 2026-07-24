//! Rectified-flow sampler, native-resolution packing, and the prompt→image / edit paths —
//! **owned by sc-14041** (native-resolution work continues in sc-14043, edit in sc-14048).
//!
//! Port of `_vendor/mage_flow/pipeline.py`. The pieces, with their pinned answers:
//!
//! - **Schedule.** `FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=6.0,
//!   use_dynamic_shifting=false)` with `set_timesteps(sigmas=linspace(1, 1/N, N))` (`:37-50`) —
//!   static shift `6σ/(1+5σ)` plus a terminal 0. **Turbo is the same ladder at N = 4**, not a
//!   distilled timestep table: N=4 gives sigmas `[1.0, 0.94736844, 0.85714287, 0.66666669, 0.0]`.
//!   The step is plain Euler, `x += (σ_next − σ_cur) · v` (`:343`).
//! - **Initial latent.** [`crate::latent`], never plain `randn`.
//! - **Packing.** Latents flatten to a variable-length token sequence (`patch_size == 1`) and are
//!   packed under a fixed budget with per-sample cumulative offsets (`cu_seqlens`) instead of
//!   block-diagonal masks. Sides must be multiples of
//!   [`SIZE_MULTIPLE`](crate::config::SIZE_MULTIPLE); the native range is
//!   [`MIN_SIZE`](crate::config::MIN_SIZE)–[`MAX_SIZE`](crate::config::MAX_SIZE) per side.
//! - **CFG.** `use_neg = cfg > 1.0` (`:326`, `:535`): at cfg ≤ 1 the reference builds **no**
//!   unconditional branch at all — one segment, one `cu_seqlens` pair, positive conditioning only.
//!   Both Turbo variants default there, so the CFG-off path is a first-class case, not an edge one.
//!   Under `batch_cfg` the duplicated uncond branch rotates at msrope frame index 1 — see
//!   [`crate::rope_embedder`].
//! - **Edit sequence.** `[noisy_target, ref_1, …, ref_N]` — **target first** (`:552-555`), which
//!   corrects the epic's `[τ, z_src, noisy z_tgt]` on both ordering and τ-placement: τ is the
//!   *separate text stream*, not part of the image sequence. Refs are clean latents,
//!   re-concatenated every step, and only the target tokens are stepped (`:557-565`). Frame index:
//!   target 0, ref_j → j. Refs are VAE-encoded at *target* resolution; the copy fed to the VL
//!   vision tower is long-edge capped at
//!   [`VL_COND_LONG_EDGE`](crate::config::VL_COND_LONG_EDGE).
//! - **Decode.** `vae.decode(unpack(tokens.float(), h, w))` (`:121-127`); `unpack` reshapes at
//!   `ceil(height/16) × ceil(width/16)` (`models/utils.py:36`).
//!
//! Boundary goldens for every one of these stages, and the hardened checker that verifies them
//! (76 invariants at cfg > 1, 71 at cfg ≤ 1, with `--self-test`), live in
//! `crates/media/mlx-gen/tools/`.
