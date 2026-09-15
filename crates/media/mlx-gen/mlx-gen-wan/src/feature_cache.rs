//! **Denoise feature caching for the dense Wan DiT** — the first of sc-18322's two approximate
//! mechanisms (epic 18304 P7), gated on
//! [`gen_core::approximation`](mlx_gen::gen_core::approximation). Its sibling,
//! [`crate::token_pruning`], is orthogonal: this module decides *which steps* run the block stack, that
//! one decides *which tokens* the stack updates, and a plan may carry both.
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
//!    [`ApproximationPlan::Approximate`](mlx_gen::gen_core::ApproximationPlan::Approximate) cannot
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
//! The captured `Δ` is itself evaluated at capture (see `TrunkCache::capture`) — an unevaluated
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
        TrunkCache::from_plan_for_test(&ApproximationPlan::feature_cache_only(policy))
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
    ///
    /// Routed through `mlx_gen::array::contiguous` because `as_slice` reads the **physical** buffer: a
    /// gather or transpose result's logical order need not match it, and comparing physical buffers
    /// would report a difference that is not one (or, worse, miss one that is).
    fn bytes(a: &Array) -> Vec<u8> {
        let a = mlx_gen::array::contiguous(&a.as_dtype(Dtype::Float32).expect("f32"))
            .expect("contiguous");
        a.eval().expect("eval");
        a.as_slice::<f32>()
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect()
    }

    /// The timesteps a 4-step schedule would hand the DiT, descending like a real trajectory.
    const TIMESTEPS: [f32; 6] = [999.0, 937.0, 833.0, 624.0, 412.0, 187.0];

    /// **Byte-identical when off, at tensor level, against a control the production code cannot reach.**
    ///
    /// An earlier revision of this test compared `forward_cached` against
    /// `forward_cached_approx(.., None, None)` and called the former "the untouched pre-sc-18322 entry
    /// point". It is not: `forward_cached` is now a one-line delegation to exactly that call, so both
    /// sides ran the same instructions and the assertion could not fail whatever either did.
    ///
    /// The control is now `forward_cached_pre_sc18322_control` — the block loop, the block body and the
    /// self-attention body transcribed verbatim from `2a42ab64c`, calling none of the sc-18322 code.
    /// Both production entry points are compared against it across a descending multi-step sequence, so
    /// a cache that leaked cross-step state, an op reordered, a bf16 cast moved, or the `ptr::eq`
    /// single-cast guard regressing into a double cast would all show up here.
    #[test]
    fn the_off_path_is_bit_identical_to_a_verbatim_pre_sc18322_control() {
        let h = harness();
        for &t in TIMESTEPS.iter() {
            let control = h
                .transformer
                .forward_cached_pre_sc18322_control(&h.latent, t, &h.cross_kv, &h.cos, &h.sin, 1)
                .expect("control forward");
            let production = h
                .transformer
                .forward_cached(&h.latent, t, &h.cross_kv, &h.cos, &h.sin, 1)
                .expect("production forward");
            let off_path = h
                .transformer
                .forward_cached_approx(&h.latent, t, &h.cross_kv, &h.cos, &h.sin, 1, None, None)
                .expect("off-path forward");
            assert_eq!(control.len(), production.len());
            assert_eq!(control.len(), off_path.len());
            for index in 0..control.len() {
                let want = bytes(&control[index]);
                assert_eq!(
                    want,
                    bytes(&production[index]),
                    "t={t} branch {index}: `forward_cached` must be byte-identical to the verbatim \
                     pre-sc-18322 forward"
                );
                assert_eq!(
                    want,
                    bytes(&off_path[index]),
                    "t={t} branch {index}: the off path must be byte-identical to the verbatim \
                     pre-sc-18322 forward"
                );
            }
        }
    }

    /// **The single bf16 cast on the un-pruned self-attention path.**
    ///
    /// `forward_split` casts its query and key/value inputs separately; on the un-pruned path they are
    /// the same tensor, and casting twice would add a duplicate full `[B, L, dim]` bf16 transient per
    /// block per step (~227 MiB at the 5B's geometry) while producing identical values — so no parity
    /// test would catch it. The `ptr::eq` guard is what prevents that, and this pins it: the two casts
    /// must be the *same array object*, not merely equal ones.
    #[test]
    fn the_unpruned_self_attention_casts_its_input_once() {
        let h = harness();
        let (tokens, _) = h
            .transformer
            .patch_embed_tokens(&h.latent)
            .expect("patch grid");
        // Exactly what `forward_split` calls, so this cannot pass while the production path regresses.
        let (xq, xkv, casts) =
            crate::transformer::split_attention_casts(&tokens, &tokens).expect("un-pruned casts");
        assert_eq!(
            casts,
            crate::transformer::AttentionCasts::Shared,
            "the un-pruned path must reuse ONE bf16 cast for the query and key/value sides"
        );
        assert_eq!(
            bytes(&xq),
            bytes(&xkv),
            "the shared cast must still deliver both sides"
        );

        // The discriminating control: a genuinely split call must cast each side, or a function that
        // always reported `Shared` would pass the assertion above for the wrong reason.
        let other = tokens.clone();
        let (_, _, casts) =
            crate::transformer::split_attention_casts(&tokens, &other).expect("split casts");
        assert_eq!(
            casts,
            crate::transformer::AttentionCasts::Separate,
            "two distinct input streams must each be cast"
        );
    }

    // ---------------------------------------------------------------------------------------------
    // Token pruning, over the same fixture and the same real block stack. It lives beside the cache's
    // suite because the two mechanisms share the harness and one of the tests below is specifically
    // about them composing.
    // ---------------------------------------------------------------------------------------------

    fn pruning(stride: u32, warmup: u32) -> mlx_gen::gen_core::TokenPruningPolicy {
        mlx_gen::gen_core::TokenPruningPolicy::new(
            mlx_gen::gen_core::TokenDropStride::new(stride).unwrap(),
        )
        .with_warmup(mlx_gen::gen_core::CacheWarmupSteps::new(warmup))
    }

    fn pruner(stride: u32, warmup: u32, total: usize) -> crate::token_pruning::TokenPruner {
        crate::token_pruning::TokenPruner::from_plan_for_test(
            &mlx_gen::gen_core::ApproximationPlan::token_pruning_only(pruning(stride, warmup)),
            total,
        )
        .expect("pruner")
        .expect("a pruning plan must yield a pruner")
    }

    /// The fixture's patchified token count.
    fn token_total(h: &Harness) -> usize {
        let (_, grid) = h
            .transformer
            .patch_embed_tokens(&h.latent)
            .expect("patch grid");
        grid.0 * grid.1 * grid.2
    }

    /// **A pruned forward differs from the exact one, and only in the dropped tokens' updates.**
    ///
    /// Two claims in one, because either alone would be satisfiable by a broken mechanism: pruning must
    /// change the prediction (or it is inert), and it must keep the prediction's shape (or nothing
    /// downstream — `apply_head`, `unpatchify`, the feature cache's shape guard — still works).
    #[test]
    fn a_pruned_forward_changes_the_prediction_and_keeps_its_shape() {
        let h = harness();
        let (_, grid) = h
            .transformer
            .patch_embed_tokens(&h.latent)
            .expect("patch grid");
        let total = grid.0 * grid.1 * grid.2;
        let mut exercised = 0usize;
        for stride in [2u32, 3, 4] {
            let declared = mlx_gen::gen_core::TokenDropStride::new(stride).unwrap();
            // The fixture's grid is tiny, so a wide stride may drop nothing at this token count. Such a
            // stride has nothing to assert here — it is exactly the exact path — so skip it rather than
            // assert a difference that cannot exist.
            if declared.kept_count(total) == total {
                continue;
            }
            exercised += 1;
            let mut pruner = pruner(stride, 0, total);
            pruner.begin_step(0);
            let exact = h
                .transformer
                .forward_cached(&h.latent, TIMESTEPS[0], &h.cross_kv, &h.cos, &h.sin, 1)
                .expect("exact forward");
            let pruned = h
                .transformer
                .forward_cached_approx(
                    &h.latent,
                    TIMESTEPS[0],
                    &h.cross_kv,
                    &h.cos,
                    &h.sin,
                    1,
                    None,
                    Some(&pruner),
                )
                .expect("pruned forward");
            assert_eq!(
                exact[0].shape(),
                pruned[0].shape(),
                "stride {stride}: a pruned forward must restore the full token count"
            );
            assert_ne!(
                bytes(&exact[0]),
                bytes(&pruned[0]),
                "stride {stride}: pruning must change the prediction — an identical result would mean \
                 the keep-set never reached the block loop"
            );
        }
        assert!(
            exercised > 0,
            "the fixture's {total}-token grid must admit at least one dropping stride"
        );
    }

    /// **A dropped token passes through a block bit-exactly.**
    ///
    /// The load-bearing property of the whole mechanism, asserted at the one place it is observable: a
    /// block's own output. Wan's residuals are `x + gate·y` and `x + y`, and the restore fills a dropped
    /// row's `y` with an exact zero, so the dropped rows of `forward_pruned`'s output must equal the
    /// dropped rows of its *input* byte for byte — while the kept rows must not (or the block did
    /// nothing at all).
    #[test]
    fn a_dropped_token_leaves_a_block_holding_exactly_the_bits_it_entered_with() {
        let h = harness();
        let (tokens, grid) = h
            .transformer
            .patch_embed_tokens(&h.latent)
            .expect("patch grid");
        let total = grid.0 * grid.1 * grid.2;
        let mut pruner = pruner(2, 0, total);
        pruner.begin_step(0);
        let keep = pruner.keep_for_block(0).expect("phase 0");

        let x_in = tokens;
        let out = h
            .transformer
            .block_forward_pruned_for_test(
                0,
                &x_in,
                TIMESTEPS[0],
                &h.cross_kv[0],
                &h.cos,
                &h.sin,
                keep,
            )
            .expect("pruned block");
        assert_eq!(
            out.shape(),
            x_in.shape(),
            "a block must restore full length"
        );

        // Compare row by row. `contiguous` first, for the same physical-buffer reason `bytes` does.
        let rows_in = row_bytes(&x_in, total);
        let rows_out = row_bytes(&out, total);
        let mut changed = 0usize;
        for index in 0..total {
            if keep.keeps(index) {
                if rows_in[index] != rows_out[index] {
                    changed += 1;
                }
            } else {
                assert_eq!(
                    rows_in[index], rows_out[index],
                    "token {index} is dropped, so it must leave the block byte-identical"
                );
            }
        }
        assert!(
            changed > 0,
            "the kept tokens must actually be updated — otherwise the pass-through assertion above is \
             vacuous"
        );
    }

    /// Per-row little-endian f32 bytes of a `[1, total, dim]` token stream.
    fn row_bytes(x: &Array, total: usize) -> Vec<Vec<u8>> {
        let flat = bytes(x);
        let per_row = flat.len() / total;
        (0..total)
            .map(|i| flat[i * per_row..(i + 1) * per_row].to_vec())
            .collect()
    }

    /// Per-row f32 values of a `[1, total, dim]` token stream.
    fn row_values(x: &Array, total: usize) -> Vec<Vec<f32>> {
        let flat = mlx_gen::array::contiguous(&x.as_dtype(Dtype::Float32).expect("f32"))
            .expect("contiguous");
        flat.eval().expect("eval");
        let flat = flat.as_slice::<f32>().to_vec();
        let per_row = flat.len() / total;
        (0..total)
            .map(|i| flat[i * per_row..(i + 1) * per_row].to_vec())
            .collect()
    }

    /// **A surviving token's update is the one the exact block would have applied — in every block.**
    ///
    /// The assertion that pins the *key/value* half of the mechanism, and the reason pruning is a
    /// bounded approximation rather than an arbitrary one: keys and values are projected from the full
    /// token stream, so a kept token's query attends over exactly the keys the exact block gives it, and
    /// its cross-attention and FFN inputs are unchanged. Prune the keys as well and a kept token sees a
    /// different key set, which moves its output far past the tolerance below.
    ///
    /// **Scope, stated precisely.** The property is per-block and conditional on the input stream: given
    /// the same `x`, block *k*'s pruned output agrees with its exact output on the kept rows. It is
    /// asserted here for **every** block of the fixture and at **every** rotation phase, not just block 0
    /// — an earlier revision checked block 0 alone, which was the only block where it could have held
    /// under the fixed keep-set that revision used. What it does *not* claim is that a pruned trajectory
    /// agrees with the exact one: dropped rows carry a one-block-stale value into the next block's keys,
    /// which is the approximation, and `no_position_is_starved_by_the_block_rotation` plus
    /// `a_pruned_step_updates_every_position` are what bound it.
    ///
    /// Not bit-exact: `Sq` changes, and MLX's fused Metal SDPA is tile-specialized by that dimension, so
    /// the same arithmetic runs in a different order. Measured worst relative |Δ| on this fixture is
    /// ~2.4e-3, squarely the bf16 class this crate's own parity suites gate at 2e-2; the bound is 1e-2
    /// relative, tight enough that a pruned key set fails it.
    #[test]
    fn a_surviving_token_gets_the_update_the_exact_block_would_have_applied() {
        let h = harness();
        let (tokens, _) = h
            .transformer
            .patch_embed_tokens(&h.latent)
            .expect("patch grid");
        let total = token_total(&h);
        let mut pruner = pruner(2, 0, total);
        let blocks = h.cross_kv.len();
        assert!(blocks > 1, "the fixture must have more than one block");

        let mut checked = 0usize;
        for step in 0..2usize {
            pruner.begin_step(step);
            for block in 0..blocks {
                let keep = pruner.keep_for_block(block).expect("phase");
                let exact = h
                    .transformer
                    .block_forward_for_test(
                        block,
                        &tokens,
                        TIMESTEPS[0],
                        &h.cross_kv[block],
                        &h.cos,
                        &h.sin,
                    )
                    .expect("exact block");
                let pruned = h
                    .transformer
                    .block_forward_pruned_for_test(
                        block,
                        &tokens,
                        TIMESTEPS[0],
                        &h.cross_kv[block],
                        &h.cos,
                        &h.sin,
                        keep,
                    )
                    .expect("pruned block");

                let exact_rows = row_values(&exact, total);
                let pruned_rows = row_values(&pruned, total);
                // Scale the tolerance to the block's own output magnitude so the bound means the same
                // thing on this fixture as it would at production scale.
                let scale = exact_rows
                    .iter()
                    .flatten()
                    .fold(0f32, |acc, v| acc.max(v.abs()))
                    .max(1e-6);
                let tolerance = scale * 1e-2;
                let mut worst = 0f32;
                for index in 0..total {
                    if !keep.keeps(index) {
                        continue;
                    }
                    checked += 1;
                    for (got, want) in pruned_rows[index].iter().zip(exact_rows[index].iter()) {
                        worst = worst.max((got - want).abs());
                    }
                }
                assert!(
                    worst <= tolerance,
                    "step {step} block {block}: a kept token must get the exact block's update (full \
                     keys/values): worst |Δ| {worst} exceeds {tolerance} (scale {scale})"
                );
            }
        }
        assert!(checked > 0, "the walk must have compared some kept rows");
    }

    /// **A pruned step updates every position** — the tensor-level refutation of starvation.
    ///
    /// The property the rotation exists for, asserted where it is observable: chain the fixture's blocks
    /// with the phases a real pruned step would use, and require that **no** token row leaves the stack
    /// holding the bits it entered with. A fixed keep-set fails this immediately — its dropped rows are
    /// still the raw patch embedding after the last block.
    #[test]
    fn a_pruned_step_updates_every_position() {
        let h = harness();
        let (tokens, _) = h
            .transformer
            .patch_embed_tokens(&h.latent)
            .expect("patch grid");
        let total = token_total(&h);
        let blocks = h.cross_kv.len();
        let mut pruner = pruner(2, 0, total);

        for step in 0..2usize {
            pruner.begin_step(step);
            let mut x = tokens.clone();
            for block in 0..blocks {
                let keep = pruner.keep_for_block(block).expect("phase");
                x = h
                    .transformer
                    .block_forward_pruned_for_test(
                        block,
                        &x,
                        TIMESTEPS[0],
                        &h.cross_kv[block],
                        &h.cos,
                        &h.sin,
                        keep,
                    )
                    .expect("pruned block");
            }
            let before = row_bytes(&tokens, total);
            let after = row_bytes(&x, total);
            for index in 0..total {
                assert_ne!(
                    before[index], after[index],
                    "step {step}: position {index} left the block stack unchanged — the rotation must \
                     give every position a contribution from at least one block"
                );
            }
        }
    }

    /// **Per-token time modulation is refused on the pruned path.**
    ///
    /// `e0..e5` would have to be gathered in lockstep with the tokens, and `apply_head` broadcasts the
    /// modulation against the token axis, so a mismatch is a silent broadcast rather than an error. The
    /// TI2V mask-blend route declares no pruning and the provider refuses a non-exact plan there; this
    /// is the local backstop, and it needs its own test because the provider-level refusal cannot reach
    /// this code.
    #[test]
    fn per_token_time_modulation_is_refused_on_the_pruned_path() {
        let h = harness();
        let (tokens, grid) = h
            .transformer
            .patch_embed_tokens(&h.latent)
            .expect("patch grid");
        let total = grid.0 * grid.1 * grid.2;
        let mut pruner = pruner(2, 0, total);
        pruner.begin_step(0);
        let keep = pruner.keep_for_block(0).expect("phase 0");
        let t_tokens = Array::from_slice(&vec![TIMESTEPS[0]; total], &[1, total as i32]);
        let error = h
            .transformer
            .block_forward_pruned_per_token_for_test(
                0,
                &tokens,
                &t_tokens,
                &h.cross_kv[0],
                &h.cos,
                &h.sin,
                keep,
            )
            .expect_err("a per-token modulation must be refused on the pruned path");
        assert!(
            error.to_string().contains("per-token time modulation"),
            "{error}"
        );

        // The scalar modulation on the same inputs succeeds, so the refusal is about the modulation
        // shape rather than about the fixture.
        h.transformer
            .block_forward_pruned_for_test(
                0,
                &tokens,
                TIMESTEPS[0],
                &h.cross_kv[0],
                &h.cos,
                &h.sin,
                keep,
            )
            .expect("a scalar modulation must be accepted");
    }

    /// **A warmup step prunes nothing, so a partially-warmed policy still perturbs only past its
    /// warmup.**
    ///
    /// Note what this test can no longer do: an earlier revision ran a policy whose warmup covered the
    /// whole trajectory and asserted a byte-identical result. That state is now **refused** at the
    /// provider bridge, because it is a plan that says `Approximate` and produces the exact output —
    /// the third encoding of *off* (see `refuse_inert_over_trajectory` and
    /// `TokenPruningPolicy::is_inert_over`). So the driver-level off proof lives in
    /// `the_public_denoise_matches_a_reference_loop_on_the_verbatim_control`, and what is left to pin
    /// here is that the warmup boundary is honoured at all: a warmup of `steps - 1` perturbs the result
    /// (one step prunes) and perturbs it *less* than no warmup at all.
    #[test]
    fn the_pruning_warmup_boundary_is_honoured_by_the_driver() {
        let h = harness();
        let cfg = tiny_cfg();
        let cancel = mlx_gen::CancelFlag::default();
        let steps = 4usize;
        let total = token_total(&h);

        let run = |prune: Option<&mut crate::token_pruning::TokenPruner>| {
            let mut ignored = |_: usize| {};
            crate::pipeline::denoise_approx(
                &h.transformer,
                crate::scheduler::SolverKind::UniPC,
                cfg.num_train_timesteps,
                steps,
                cfg.sample_shift,
                1.0,
                &h.ctx,
                None,
                &h.latent,
                &cancel,
                &mut ignored,
                None,
                prune,
            )
            .expect("denoise")
        };

        let exact = run(None);
        let mut late = pruner(2, steps as u32 - 1, total);
        let late = run(Some(&mut late));
        let mut early = pruner(2, 0, total);
        let early = run(Some(&mut early));

        assert_eq!(exact.shape(), late.shape());
        assert_ne!(
            bytes(&exact),
            bytes(&late),
            "a warmup of steps-1 still leaves one pruned step, so the result must move"
        );
        assert_ne!(
            bytes(&late),
            bytes(&early),
            "pruning every step must differ from pruning only the last one — the warmup boundary is \
             load-bearing, not decorative"
        );

        // And the contract agrees with the driver about which of these is inert.
        assert!(!pruning(2, steps as u32 - 1).is_inert_over(steps));
        assert!(pruning(2, steps as u32).is_inert_over(steps));
    }

    /// **An approximate policy that engages on no step of THIS trajectory is refused.**
    ///
    /// The last door to a "second encoding of off": a warmup inside the declared `max_warmup_steps` can
    /// still cover a short trajectory, yielding an `Approximate` plan whose result is bit-identical to
    /// the exact path. The step count is only known at the provider, so the provider is where it is
    /// refused — and the refusal must name which mechanism was inert, since a plan may carry both.
    #[test]
    fn a_policy_inert_over_this_trajectory_is_refused_at_the_provider_bridge() {
        use mlx_gen::gen_core::{ApproximationPlan, CacheWarmupSteps};

        let inert_cache = ApproximationPlan::feature_cache_only(
            FeatureCachePolicy::new(CacheReuseInterval::new(2).unwrap())
                .with_warmup(CacheWarmupSteps::new(8)),
        );
        let error = crate::model::refuse_inert_over_trajectory("prov", &inert_cache, 4)
            .expect_err("a warmup of 8 over 4 steps reuses nothing");
        let text = error.to_string();
        assert!(text.contains("denoise feature cache"), "{text}");
        assert!(text.contains("reuses nothing"), "{text}");
        assert!(text.contains("4 steps"), "{text}");

        let inert_prune = ApproximationPlan::token_pruning_only(pruning(2, 8));
        let text = crate::model::refuse_inert_over_trajectory("prov", &inert_prune, 4)
            .expect_err("a warmup of 8 over 4 steps prunes nothing")
            .to_string();
        assert!(text.contains("token pruning"), "{text}");
        assert!(text.contains("prunes nothing"), "{text}");

        // The discriminating controls: the same policies over a longer trajectory are admitted, and the
        // exact plan is always admitted.
        crate::model::refuse_inert_over_trajectory("prov", &inert_cache, 20)
            .expect("20 steps reach past the warmup");
        crate::model::refuse_inert_over_trajectory("prov", &inert_prune, 20)
            .expect("20 steps reach past the warmup");
        crate::model::refuse_inert_over_trajectory("prov", &ApproximationPlan::Exact, 1)
            .expect("the exact plan is never inert-refused");
    }

    /// **Cancellation is observed on a pruned step, and progress does not skip one.**
    ///
    /// Pruning changes what a step computes, not how a step is committed — the cancel check, the
    /// per-step `eval` and the progress callback are the same straight-line statements outside the
    /// branch. Asserted against both mechanisms at once, since a step can be reused *and* pruned.
    #[test]
    fn cancellation_and_progress_survive_a_step_that_both_reuses_and_prunes() {
        let h = harness();
        let cfg = tiny_cfg();
        let cancel = mlx_gen::CancelFlag::default();
        let mut trunk = cache(2, 0);
        let mut prune = pruner(2, 0, token_total(&h));
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
            Some(&mut prune),
        )
        .expect_err("a cancelled denoise must not return a latent");
        assert!(matches!(error, Error::Canceled), "{error:?}");
        assert_eq!(
            completed,
            vec![1, 2, 3],
            "the loop must stop at the step following the cancellation, reuse and pruning included"
        );
    }

    /// **The two mechanisms compose**, and each contributes: cache+prune differs from either alone.
    #[test]
    fn the_cache_and_pruning_compose_into_a_third_distinct_trajectory() {
        let h = harness();
        let cfg = tiny_cfg();
        let cancel = mlx_gen::CancelFlag::default();
        let run = |trunk: Option<&mut TrunkCache>,
                   prune: Option<&mut crate::token_pruning::TokenPruner>| {
            let mut ignored = |_: usize| {};
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
                &mut ignored,
                trunk,
                prune,
            )
            .expect("denoise")
        };
        let total = token_total(&h);
        let exact = run(None, None);
        let cache_only = run(Some(&mut cache(2, 0)), None);
        let prune_only = run(None, Some(&mut pruner(2, 0, total)));
        let both = run(Some(&mut cache(2, 0)), Some(&mut pruner(2, 0, total)));

        for (name, latents) in [
            ("cache", &cache_only),
            ("prune", &prune_only),
            ("both", &both),
        ] {
            assert_eq!(exact.shape(), latents.shape(), "{name}: shape preserved");
            assert_ne!(
                bytes(&exact),
                bytes(latents),
                "{name}: must change the result"
            );
        }
        assert_ne!(
            bytes(&cache_only),
            bytes(&both),
            "adding pruning on top of the cache must change the result"
        );
        assert_ne!(
            bytes(&prune_only),
            bytes(&both),
            "adding the cache on top of pruning must change the result"
        );
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
                    None,
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

    /// **`denoise` is byte-identical to a reference loop built on the verbatim control.**
    ///
    /// The driver-level half of the off proof. The loop below is the pre-sc-18322 six lines rebuilt in
    /// the test, and — unlike an earlier revision — its forward is
    /// `forward_cached_pre_sc18322_control`, so neither side of the comparison runs any sc-18322 code
    /// in common. It pins both the driver's operation order and the transformer's.
    #[test]
    fn the_public_denoise_matches_a_reference_loop_on_the_verbatim_control() {
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
                .forward_cached_pre_sc18322_control(&latents, t, &cross_kv, &cos, &sin, 1)
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
            None,
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
                None,
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
            None,
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

    /// The retained residual holds the right values.
    ///
    /// Deliberately **not** titled as a proof of the evaluation barrier. `capture`'s
    /// `eval` is load-bearing for residency — an unevaluated residual is a graph node still referencing
    /// the block weights, so rung 4's windowed materialization would free nothing — but MLX exposes no
    /// public way to ask whether an array is materialized, and `as_slice` evaluates implicitly, so
    /// deleting that `eval` is a mutation no test in this workspace can catch. The same is true of the
    /// two identical barriers already in `transformer.rs` and `block_stream.rs`, which are held by
    /// comment for the same reason. What this test does pin is the residual's contents.
    #[test]
    fn the_retained_residual_holds_the_captured_delta() {
        let mut cache = cache(2, 0);
        let x_in = Array::from_slice(&[1.0f32, 2.0], &[2]);
        let x_out = Array::from_slice(&[4.0f32, 6.0], &[2]);
        cache.capture(&x_in, &x_out).unwrap();
        let retained = cache.retained.as_ref().unwrap();
        assert_eq!(retained.as_slice::<f32>(), &[3.0, 4.0]);
    }
}
