//! **Denoise feature caching for the dense Wan DiT** — the approximate-capability mechanism of
//! sc-18322 (epic 18304 P7), gated on [`gen_core::approximation`](mlx_gen::gen_core::approximation).
//!
//! # What is cached, and why *this* quantity
//!
//! The Wan DiT is a single-stream transformer: the block stack carries one tensor `x` of stable shape
//! `[B, L, dim]` from `patch_embedding` to `head`, and each of its blocks adds three residuals to it
//! without ever changing its shape (`transformer::Block::forward`). Its **trunk residual**
//!
//! ```text
//! Δ = x_after_the_block_stack − x_entering_the_block_stack
//! ```
//!
//! is therefore a single well-formed tensor that captures the entire aggregate contribution of every
//! block at one denoise step. Successive denoise steps evaluate the same stack at adjacent noise
//! levels, so consecutive `Δ`s are highly correlated — which is what makes reusing one across a step
//! a cheap approximation rather than a broken render.
//!
//! Two Wan-specific facts make the trunk residual the *right* granularity, not merely a convenient
//! one:
//!
//! * **Everything step-invariant is already hoisted out of the block stack.** Cross-attention K/V are
//!   built once per generate from the text context ([`crate::pipeline`]'s `StepCache`) and carry no
//!   RoPE, and the RoPE tables are fixed by the patch grid. Within one step the only inputs that vary
//!   are `x` and the shared time modulation `e0`, so "the stack's contribution at this step" is a
//!   clean function of the step rather than of hidden per-block state.
//! * **Per-block caching does not fit the video lane.** At production geometry the 5B carries
//!   `L = 18480` tokens × `dim = 3072`, so one f32 `[1, L, dim]` tensor is ~227 MiB. Retaining one per
//!   block across a step would be ~6.8 GiB on a 30-block stack (and double under batched CFG) —
//!   spending the whole memory ladder's headroom to save arithmetic, on the one lane where memory is
//!   the binding constraint. The trunk residual is **one** such tensor.
//!
//! The `head` is deliberately *not* cached. It consumes the per-step modulation `e` and is two
//! elementwise ops plus one projection, so re-running it every step costs almost nothing and keeps the
//! step's own timestep fully honoured even on a reused step.
//!
//! # Byte-identical when off
//!
//! [`TrunkCache`] is threaded as an `Option<&mut TrunkCache>`. With `None` — the state every
//! production caller is in, and the only state the pre-existing entry points
//! ([`WanTransformer::forward_cached`](crate::transformer::WanTransformer::forward_cached),
//! [`crate::pipeline::denoise`]) can produce — the block loop runs the identical instruction sequence
//! it ran before this module existed, and nothing here is constructed, evaluated or branched on beyond
//! one `is_some`. There is no "cache configured to do nothing": see
//! [`gen_core::approximation`](mlx_gen::gen_core::approximation) for why `off` is absence rather than
//! a parameter value.
//!
//! # Not reachable from a request
//!
//! The mechanism is real, tested code, and no request can select it. Two independent reasons:
//!
//! 1. the contract refuses every approximate selection until a quality-characterization artifact
//!    family exists, and the binding that would admit one is uninhabited; and
//! 2. [`TrunkCache`]'s only constructor is `#[cfg(test)]`, so even a caller holding an
//!    [`ApproximationPlan::FeatureCache`](mlx_gen::gen_core::ApproximationPlan::FeatureCache) cannot
//!    turn it into a cache outside this crate's own unit tests.
//!
//! # Cancel and evaluation discipline
//!
//! The hazard a step-skipping mechanism creates is that skipping the compute also skips the
//! *evaluation barrier* that makes MLX's lazy graph materialize — and with it the per-step cancel
//! check, since an un-materialized graph makes every cancel check pass until VAE decode
//! (`pipeline::denoise`'s own comment says exactly this).
//!
//! The discipline here is structural rather than defensive: the reuse decision lives **three call
//! levels below** the step loop, inside
//! [`WanTransformer::forward_with_modulation`](crate::transformer::WanTransformer), while the cancel
//! check, the per-step `eval`, and the progress callback are straight-line statements in
//! [`crate::pipeline::denoise_approx`]'s loop body, outside any branch this module can influence. A
//! reused step therefore performs exactly the same cancel check, the same `eval` and the same progress
//! emission as a recomputed one, because there is no code path in which it does not. The unit tests
//! below pin both halves: cancellation requested at a *reuse* step is observed at that step, and the
//! progress callback fires once per step for every step regardless of reuse.
//!
//! The captured `Δ` is itself evaluated at capture (see [`TrunkCache::capture`]) — an unevaluated
//! residual would be a graph node still referencing the block weights, which is the hazard
//! `transformer.rs`'s windowed-forward barrier exists to prevent, and a cache that reintroduced it
//! would silently defeat memory-ladder rung 4.

use mlx_rs::ops::add;
use mlx_rs::Array;

use mlx_gen::gen_core::FeatureCachePolicy;
use mlx_gen::Error;

type Result<T> = std::result::Result<T, Error>;

/// The per-generate denoise feature cache for one dense Wan DiT: the retained trunk residual plus the
/// step the driver is currently on.
///
/// One cache belongs to one denoise loop over one transformer. It must **not** outlive an expert swap
/// (the MoE A14B paths reload a different transformer mid-trajectory, so a residual captured under one
/// expert is meaningless under the other) — which is why those routes refuse a non-exact plan by name
/// instead of threading a cache; see [`crate::pipeline::refuse_unwired_approximation`].
///
/// There is no public constructor outside `cfg(test)`. That is the second, independent reason a
/// production request cannot reach the approximate path — see the module docs.
#[derive(Debug)]
pub struct TrunkCache {
    policy: FeatureCachePolicy,
    /// The step index the driver declared via [`begin_step`](Self::begin_step). `None` before the
    /// first one, which is an error rather than an implicit step 0: a driver that forgets to declare
    /// the step would otherwise silently treat every forward as step 0 and never reuse anything.
    step: Option<usize>,
    /// The retained trunk residual `[B, L, dim]` (f32, evaluated), with the shape it was captured at.
    retained: Option<Array>,
}

impl TrunkCache {
    /// Build the cache a plan asks for — `None` for [`ApproximationPlan::Exact`], which is the only
    /// plan the contract can produce today.
    ///
    /// `#[cfg(test)]` on purpose. The contract already refuses every approximate selection; this makes
    /// the *mechanism* unreachable from production code as well, so the uncharacterized path cannot be
    /// entered even by a caller inside this crate that constructs a plan by hand.
    #[cfg(test)]
    pub(crate) fn from_plan_for_test(
        plan: &mlx_gen::gen_core::ApproximationPlan,
    ) -> Option<TrunkCache> {
        plan.feature_cache().map(|policy| TrunkCache {
            policy: *policy,
            step: None,
            retained: None,
        })
    }

    /// The policy this cache runs.
    pub fn policy(&self) -> &FeatureCachePolicy {
        &self.policy
    }

    /// Declare the denoise step the next forward belongs to.
    ///
    /// Called by the step loop, which is the only place that knows the step index — the transformer
    /// forward sees only a timestep value, and deriving a step index from it would break the moment a
    /// multi-evaluation solver called the same timestep twice. Keeping the index with the driver is
    /// also what keeps this mechanism honestly scoped to the one-forward-per-step native loop.
    pub fn begin_step(&mut self, index: usize) {
        self.step = Some(index);
    }

    /// Whether the current step recomputes the trunk (as opposed to reusing the retained residual).
    ///
    /// Delegates the phase decision to the contract's single decision point, so this crate cannot
    /// disagree with the declared policy about which steps are approximate. A step with no retained
    /// residual always recomputes even if the policy would reuse — belt for the braces of
    /// [`CacheReuseInterval::recomputes_step`](mlx_gen::gen_core::CacheReuseInterval::recomputes_step)
    /// guaranteeing that step 0 primes.
    pub(crate) fn recomputes(&self) -> Result<bool> {
        let step = self.step.ok_or_else(|| {
            Error::Msg(
                "wan: denoise feature cache used before the step loop declared a step index; call \
                 TrunkCache::begin_step once per denoise step"
                    .into(),
            )
        })?;
        Ok(self.policy.recomputes_step(step) || self.retained.is_none())
    }

    /// Reuse the retained trunk residual on top of this step's stack input.
    ///
    /// `x_in` is the token stream entering the block stack (bf16, possibly a CFG broadcast view); the
    /// residual is f32, so the sum is f32 — exactly the dtype the exact path produces, where the first
    /// block's f32 adaLN gate promotes the bf16 stream on the very first residual.
    pub(crate) fn reuse(&self, x_in: &Array) -> Result<Array> {
        let retained = self.retained.as_ref().ok_or_else(|| {
            Error::Msg(
                "wan: denoise feature cache has no retained trunk residual to reuse; the first step \
                 of every policy must recompute"
                    .into(),
            )
        })?;
        // A residual captured at a different geometry or CFG width is not applicable to this step.
        // Refusing is the only safe answer: MLX would happily broadcast a `[1, L, dim]` residual over
        // a `[2, L, dim]` stream and silently render the cond branch's correction into both.
        if retained.shape() != x_in.shape() {
            return Err(Error::Msg(format!(
                "wan: denoise feature cache retained a {:?} trunk residual but this step's token \
                 stream is {:?}; a cache must not outlive the geometry or CFG width it was captured at",
                retained.shape(),
                x_in.shape()
            )));
        }
        Ok(add(retained, x_in)?)
    }

    /// Retain this step's trunk residual for later reuse.
    ///
    /// The residual is **evaluated here**. An unevaluated `Δ` would be a lazy graph node still
    /// referencing every block's weights, so retaining one across steps would keep the whole stack
    /// alive and silently defeat memory-ladder rung 4's windowed materialization — the same trap
    /// `transformer.rs`'s windowed barrier documents. Forcing it at capture also means the reuse step
    /// adds two materialized tensors instead of replaying a graph.
    pub(crate) fn capture(&mut self, x_in: &Array, x_out: &Array) -> Result<()> {
        let delta = mlx_rs::ops::subtract(x_out, x_in)?;
        mlx_rs::transforms::eval([&delta])?;
        self.retained = Some(delta);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::gen_core::{
        ApproximationPlan, CacheReuseInterval, CacheWarmupSteps, FeatureCachePolicy,
    };
    use mlx_rs::Dtype;

    fn cache(interval: u32, warmup: u32) -> TrunkCache {
        let policy = FeatureCachePolicy::new(CacheReuseInterval::new(interval).unwrap())
            .with_warmup(CacheWarmupSteps::new(warmup));
        TrunkCache::from_plan_for_test(&ApproximationPlan::FeatureCache(policy))
            .expect("a feature-cache plan must yield a cache")
    }

    // ---------------------------------------------------------------------------------------------
    // The tensor-level half: the checked-in 2-block S5 fixture, driven through the REAL block stack.
    //
    // Weights-free in the sense the crate's other mechanism gates are (`block_stream.rs`): no real
    // checkpoint, no accelerator campaign — a 128-dim / 2-block / 1-head DiT small enough to run in
    // CI, but the same `Block::forward` and the same `forward_with_modulation` body production runs.
    // ---------------------------------------------------------------------------------------------

    fn tiny_cfg() -> crate::config::WanModelConfig {
        let mut c = crate::config::WanModelConfig::wan21_t2v_1_3b();
        c.dim = 128;
        c.num_heads = 1;
        c.num_layers = 2;
        c.ffn_dim = 256;
        c.freq_dim = 256;
        c.text_dim = 32;
        c.text_len = 8;
        c.in_dim = 16;
        c.out_dim = 16;
        c.vae_z_dim = 16;
        c
    }

    fn fixture() -> mlx_gen::weights::Weights {
        let path = format!(
            "{}/tests/fixtures/s5_low.safetensors",
            env!("CARGO_MANIFEST_DIR")
        );
        mlx_gen::weights::Weights::from_file(&path).unwrap_or_else(|e| panic!("read {path}: {e}"))
    }

    /// One loaded tiny transformer plus the per-generate caches the denoise loop builds, so a test
    /// drives exactly the seam `pipeline::denoise_approx` drives.
    struct Harness {
        transformer: crate::WanTransformer,
        cross_kv: Vec<(Array, Array)>,
        cos: Array,
        sin: Array,
        latent: Array,
        /// The **embedded** conditioning, which is what `pipeline::denoise` takes.
        ctx: Array,
    }

    fn harness() -> Harness {
        let cfg = tiny_cfg();
        let weights = fixture();
        let transformer =
            crate::WanTransformer::from_weights(&weights, &cfg).expect("tiny transformer");
        let raw = weights.require("ctx_cond").expect("ctx_cond").clone();
        let ctx = transformer.embed_text(&raw).expect("embed_text");
        let cross_kv = transformer.prepare_cross_kv(&ctx).expect("cross kv");
        let latent = weights.require("init_noise").expect("init_noise").clone();
        let grid = transformer.patch_grid(&latent);
        let (cos, sin) = transformer.prepare_rope(grid).expect("rope");
        Harness {
            transformer,
            cross_kv,
            cos,
            sin,
            latent,
            ctx,
        }
    }

    /// Raw little-endian f32 bytes of an evaluated array — bit-identity, not a tolerance.
    fn bytes(a: &Array) -> Vec<u8> {
        let a = a.as_dtype(Dtype::Float32).expect("f32");
        a.eval().expect("eval");
        a.as_slice::<f32>()
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect()
    }

    /// The timesteps a 4-step schedule would hand the DiT, descending like a real trajectory.
    const TIMESTEPS: [f32; 6] = [999.0, 937.0, 833.0, 624.0, 412.0, 187.0];

    /// **Byte-identical when off, at tensor level.**
    ///
    /// The off path must be the provider's exact path, and this is the proof that does not reduce to
    /// "the same function returns the same thing": `forward_cached` is the untouched pre-sc-18322 entry
    /// point, `forward_cached_approx(.., None)` is the new one carrying the cache parameter, and every
    /// step of a multi-step sequence must agree **bit for bit** across both. A cache that leaked
    /// cross-step state, reordered an op, or perturbed the bf16→f32 promotion at the first residual
    /// would fail here even with the policy absent.
    #[test]
    fn the_off_path_is_bit_identical_to_the_pre_existing_forward() {
        let h = harness();
        for &t in TIMESTEPS.iter() {
            let expected = h
                .transformer
                .forward_cached(&h.latent, t, &h.cross_kv, &h.cos, &h.sin, 1)
                .expect("exact forward");
            let got = h
                .transformer
                .forward_cached_approx(&h.latent, t, &h.cross_kv, &h.cos, &h.sin, 1, None)
                .expect("off-path forward");
            assert_eq!(expected.len(), got.len());
            for (index, (a, b)) in expected.iter().zip(got.iter()).enumerate() {
                assert_eq!(
                    bytes(a),
                    bytes(b),
                    "t={t} branch {index}: the off path must be byte-identical to the exact forward"
                );
            }
        }
    }

    /// **The mechanism is real, and only where the policy says.**
    ///
    /// A recomputed step must be bit-identical to the exact forward at that step — nothing about
    /// carrying a cache may perturb the steps it does not skip — while a reused step must *differ*,
    /// because a reuse that happened to match would mean the cache was silently recomputing and the
    /// whole mechanism was inert. Shapes are asserted on both; nothing here counts elements.
    #[test]
    fn a_recompute_step_matches_the_exact_forward_and_a_reused_step_does_not() {
        let h = harness();
        let mut cache = cache(2, 1);
        // interval 2, warmup 1 ⇒ steps 0,1 recompute (warmup + priming), then 3,5 recompute and 2,4
        // reuse. Derived from the contract's own decision point rather than restated here, so the two
        // cannot drift apart.
        let mut reused_steps = 0usize;
        let mut recomputed_steps = 0usize;
        for (step, &t) in TIMESTEPS.iter().enumerate() {
            let exact = h
                .transformer
                .forward_cached(&h.latent, t, &h.cross_kv, &h.cos, &h.sin, 1)
                .expect("exact forward");
            cache.begin_step(step);
            let recomputes = cache.recomputes().expect("a declared step must decide");
            let got = h
                .transformer
                .forward_cached_approx(
                    &h.latent,
                    t,
                    &h.cross_kv,
                    &h.cos,
                    &h.sin,
                    1,
                    Some(&mut cache),
                )
                .expect("cached forward");
            assert_eq!(
                exact[0].shape(),
                got[0].shape(),
                "step {step}: a reused step must keep the prediction's shape"
            );
            if recomputes {
                recomputed_steps += 1;
                assert_eq!(
                    bytes(&exact[0]),
                    bytes(&got[0]),
                    "step {step} recomputes, so it must be bit-identical to the exact forward"
                );
            } else {
                reused_steps += 1;
                assert_ne!(
                    bytes(&exact[0]),
                    bytes(&got[0]),
                    "step {step} reuses a residual captured at a different timestep, so it must \
                     differ from the exact forward — an equal result would mean the cache is inert"
                );
            }
        }
        assert!(
            reused_steps > 0 && recomputed_steps > 0,
            "the trajectory must exercise both phases (reused {reused_steps}, recomputed \
             {recomputed_steps})"
        );
    }

    /// **`denoise` is byte-identical to a reference loop that never touches this module.**
    ///
    /// The independent control for the off path at *driver* level: the loop below is the pre-sc-18322
    /// six lines, rebuilt in the test out of nothing but public API (`make_scheduler` + the untouched
    /// `forward_cached` + `WanScheduler::step`). If threading the cache through `denoise_approx`
    /// changed the order of a single operation, the final latents would diverge.
    #[test]
    fn the_public_denoise_matches_an_independent_reference_loop() {
        use crate::scheduler::{make_scheduler, SolverKind};
        let h = harness();
        let cfg = tiny_cfg();
        let steps = 4;
        let cancel = mlx_gen::CancelFlag::default();

        let mut ignored = |_: usize| {};
        let produced = crate::pipeline::denoise(
            &h.transformer,
            SolverKind::UniPC,
            cfg.num_train_timesteps,
            steps,
            cfg.sample_shift,
            1.0,
            &h.ctx,
            None,
            &h.latent,
            &cancel,
            &mut ignored,
        )
        .expect("denoise");

        let mut sched = make_scheduler(SolverKind::UniPC, cfg.num_train_timesteps);
        sched.set_timesteps(steps, cfg.sample_shift);
        let timesteps: Vec<f32> = sched.timesteps().to_vec();
        let cross_kv = h.transformer.prepare_cross_kv(&h.ctx).expect("cross kv");
        let grid = h.transformer.patch_grid(&h.latent);
        let (cos, sin) = h.transformer.prepare_rope(grid).expect("rope");
        let mut latents = h.latent.clone();
        for &t in timesteps.iter() {
            let preds = h
                .transformer
                .forward_cached(&latents, t, &cross_kv, &cos, &sin, 1)
                .expect("reference forward");
            latents = sched.step(&preds[0], &latents).expect("reference step");
            mlx_rs::transforms::eval([&latents]).expect("eval");
        }

        assert_eq!(
            bytes(&produced),
            bytes(&latents),
            "denoise must be byte-identical to the reference loop it delegates to"
        );
    }

    /// **Cancellation requested at a *reused* step is still observed at that step.**
    ///
    /// The lazy-eval-defeats-cancel hazard, aimed at exactly the step where the mechanism skips work.
    /// With interval 2 and no warmup, steps 0/2/4 recompute and 1/3/5 reuse; the flag is raised from
    /// the progress callback after step 3 completes (`on_step(3)`, i.e. loop index 2), so the next
    /// iteration is index 3 — a **reuse** step. It must return `Canceled`, and exactly three steps must
    /// have completed. A mechanism that moved the cancel check or the per-step `eval` inside the
    /// recompute branch would run to completion here instead.
    #[test]
    fn cancellation_is_observed_at_a_reused_step() {
        let h = harness();
        let cfg = tiny_cfg();
        let mut trunk = cache(2, 0);
        let cancel = mlx_gen::CancelFlag::default();
        let mut completed = Vec::new();
        let mut on_step = |i: usize| {
            completed.push(i);
            if i == 3 {
                cancel.cancel();
            }
        };
        let error = crate::pipeline::denoise_approx(
            &h.transformer,
            crate::scheduler::SolverKind::UniPC,
            cfg.num_train_timesteps,
            6,
            cfg.sample_shift,
            1.0,
            &h.ctx,
            None,
            &h.latent,
            &cancel,
            &mut on_step,
            Some(&mut trunk),
        )
        .expect_err("a cancelled denoise must not return a latent");
        assert!(
            matches!(error, Error::Canceled),
            "cancellation must be the typed variant, not a stringified failure: {error:?}"
        );
        assert_eq!(
            completed,
            vec![1, 2, 3],
            "the loop must stop at the reuse step following the cancellation"
        );
    }

    /// **Every step reports progress and is materialized, reused or not.**
    ///
    /// `denoise_approx` emits progress immediately after its unconditional per-step `eval`, in the same
    /// straight-line block outside the cache branch — so a progress emission per step is also an
    /// evaluation barrier per step. Asserting the emissions is therefore how the barrier's cadence is
    /// pinned without an instrumentation hook: a mechanism that skipped the tail of the loop body on a
    /// reused step would drop both together.
    #[test]
    fn progress_and_the_per_step_barrier_do_not_skip_a_reused_step() {
        let h = harness();
        let cfg = tiny_cfg();
        let cancel = mlx_gen::CancelFlag::default();
        for warmup in [0u32, 1] {
            let mut trunk = cache(2, warmup);
            let mut completed = Vec::new();
            let mut on_step = |i: usize| completed.push(i);
            crate::pipeline::denoise_approx(
                &h.transformer,
                crate::scheduler::SolverKind::UniPC,
                cfg.num_train_timesteps,
                5,
                cfg.sample_shift,
                1.0,
                &h.ctx,
                None,
                &h.latent,
                &cancel,
                &mut on_step,
                Some(&mut trunk),
            )
            .expect("cached denoise");
            assert_eq!(
                completed,
                vec![1, 2, 3, 4, 5],
                "warmup {warmup}: every step must report once, reused steps included"
            );
        }
    }

    /// **The cached trajectory is a different trajectory** — the approximation is not cosmetic — and it
    /// is still a well-formed latent of the exact path's shape.
    ///
    /// Shape, never element counts: the assertion that matters is that reuse changed the result while
    /// leaving the tensor usable by the VAE decode that follows.
    #[test]
    fn the_cached_trajectory_differs_from_the_exact_one_and_keeps_its_shape() {
        let h = harness();
        let cfg = tiny_cfg();
        let cancel = mlx_gen::CancelFlag::default();
        let mut ignored = |_: usize| {};
        let exact = crate::pipeline::denoise(
            &h.transformer,
            crate::scheduler::SolverKind::UniPC,
            cfg.num_train_timesteps,
            5,
            cfg.sample_shift,
            1.0,
            &h.ctx,
            None,
            &h.latent,
            &cancel,
            &mut ignored,
        )
        .expect("exact denoise");

        let mut trunk = cache(2, 0);
        let mut ignored = |_: usize| {};
        let cached = crate::pipeline::denoise_approx(
            &h.transformer,
            crate::scheduler::SolverKind::UniPC,
            cfg.num_train_timesteps,
            5,
            cfg.sample_shift,
            1.0,
            &h.ctx,
            None,
            &h.latent,
            &cancel,
            &mut ignored,
            Some(&mut trunk),
        )
        .expect("cached denoise");

        assert_eq!(
            exact.shape(),
            cached.shape(),
            "a cached trajectory must still produce the exact path's latent shape"
        );
        assert_ne!(
            bytes(&exact),
            bytes(&cached),
            "reusing a trunk residual must change the trajectory — an identical result would mean \
             the mechanism never engaged"
        );
    }

    #[test]
    fn an_exact_plan_yields_no_cache_at_all() {
        // The mechanism's half of byte-identical-when-off: `Exact` does not produce a configured-inert
        // cache that the forward would then walk, it produces nothing to walk.
        assert!(TrunkCache::from_plan_for_test(&ApproximationPlan::Exact).is_none());
    }

    #[test]
    fn a_forward_before_the_driver_declares_a_step_is_refused() {
        // Not a theoretical guard: without it a driver that forgot `begin_step` would silently see
        // step 0 forever, recompute every step, and report a working cache that never caches.
        let cache = cache(2, 0);
        let error = cache
            .recomputes()
            .expect_err("a cache with no declared step must refuse");
        assert!(error.to_string().contains("begin_step"), "{error}");
    }

    #[test]
    fn the_priming_step_recomputes_and_the_policy_decides_the_rest() {
        let mut cache = cache(2, 1);
        // Warmup step 0 and priming step 1 recompute; then every other step.
        let mut decisions = Vec::new();
        for step in 0..6 {
            cache.begin_step(step);
            decisions.push(cache.recomputes().unwrap());
            if decisions[step] {
                // Stand in for a recompute so the retained-residual belt does not mask the policy.
                cache.retained = Some(Array::from_slice(&[1.0f32, 2.0], &[2]));
            }
        }
        assert_eq!(decisions, vec![true, true, false, true, false, true]);
    }

    #[test]
    fn a_reuse_step_with_no_retained_residual_recomputes_rather_than_failing() {
        // The belt: even if a policy's phase said "reuse" at a step with an empty cache, the answer is
        // recompute. `reuse` still refuses loudly if it is ever called in that state, because the two
        // guards protect against different mistakes.
        let mut cache = cache(2, 0);
        cache.begin_step(1);
        assert!(
            cache.recomputes().unwrap(),
            "an empty cache must recompute regardless of phase"
        );
        let x = Array::from_slice(&[1.0f32, 2.0], &[2]);
        let error = cache
            .reuse(&x)
            .expect_err("an empty cache cannot be reused");
        assert!(error.to_string().contains("must recompute"), "{error}");
    }

    #[test]
    fn a_captured_residual_reconstructs_the_step_it_was_captured_from() {
        // The mechanism's arithmetic, at tensor level: Δ = out − in, and in + Δ == out exactly. Not a
        // tautology to assert — it pins the sign convention and the dtype promotion, and a mutation
        // that swaps `subtract`'s operands or adds the OUTPUT instead of the residual fails here.
        let mut cache = cache(2, 0);
        let x_in = Array::from_slice(&[1.0f32, -2.0, 3.5, 0.0], &[1, 2, 2])
            .as_dtype(Dtype::Bfloat16)
            .unwrap();
        let x_out = Array::from_slice(&[2.0f32, -1.0, 7.0, 0.5], &[1, 2, 2]);
        cache.capture(&x_in, &x_out).unwrap();

        let reused = cache.reuse(&x_in).unwrap();
        assert_eq!(reused.shape(), x_out.shape(), "shape must be preserved");
        assert_eq!(
            reused.dtype(),
            Dtype::Float32,
            "a bf16 stream plus an f32 residual must promote exactly as the first block's f32 gate does"
        );
        let got: Vec<f32> = reused.as_dtype(Dtype::Float32).unwrap().as_slice().to_vec();
        let want: Vec<f32> = x_out.as_slice().to_vec();
        assert_eq!(got, want, "in + (out − in) must be exactly out");
    }

    #[test]
    fn a_residual_from_a_different_geometry_or_cfg_width_is_refused() {
        let mut cache = cache(2, 0);
        let x_in = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[1, 2, 2]);
        let x_out = Array::from_slice(&[2.0f32, 4.0, 6.0, 8.0], &[1, 2, 2]);
        cache.capture(&x_in, &x_out).unwrap();

        // A CFG-batched stream against a B=1 residual: MLX would broadcast it silently, rendering the
        // cond branch's correction into the uncond branch too.
        let batched = Array::from_slice(&[1.0f32; 8], &[2, 2, 2]);
        let error = cache
            .reuse(&batched)
            .expect_err("a B=1 residual must not be broadcast over a B=2 stream");
        assert!(error.to_string().contains("CFG width"), "{error}");

        // And a different token count.
        let longer = Array::from_slice(&[1.0f32; 6], &[1, 3, 2]);
        assert!(cache.reuse(&longer).is_err(), "L must match");
    }

    #[test]
    fn the_retained_residual_is_materialized_at_capture() {
        // The rung-4 interaction: a retained residual that is still a lazy graph node keeps the block
        // weights alive. `capture` evaluates it, so reading it back needs no further eval — which is
        // what `as_slice` on an unevaluated array could not give.
        let mut cache = cache(2, 0);
        let x_in = Array::from_slice(&[1.0f32, 2.0], &[2]);
        let x_out = Array::from_slice(&[4.0f32, 6.0], &[2]);
        cache.capture(&x_in, &x_out).unwrap();
        let retained = cache.retained.as_ref().unwrap();
        assert_eq!(retained.as_slice::<f32>(), &[3.0, 4.0]);
    }
}
