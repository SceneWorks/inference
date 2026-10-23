//! Full closed-request epoch peak under the explicit native owner-retirement
//! contract. Never selected from an enum/observer or ordinary synchronization.
//! No anchor subtraction, donation, process mapping or externally supplied DAG.
use super::*;
#[path = "native_liveness.rs"]
mod live;
use live::{LiveLatent, LiveOps};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Evaluation {
    NativeAllocation,
    EpochCountOnly,
}
struct RetiredProfile<'a> {
    facts: &'a SanaPricingFacts,
    plan: &'a SanaPricingPlan,
    page: u64,
    backend: NativeBackend,
    peak: Cell<u64>,
    counts: std::cell::RefCell<NativeEpochCounts>,
    evaluation: Evaluation,
}
use std::cell::Cell;
impl<'a> RetiredProfile<'a> {
    fn new(
        facts: &'a SanaPricingFacts,
        plan: &'a SanaPricingPlan,
        page: u64,
        backend: NativeBackend,
    ) -> Result<Self> {
        NativeLedger::new(page, backend)?;
        Ok(Self {
            facts,
            plan,
            page,
            backend,
            peak: Cell::new(0),
            counts: std::cell::RefCell::new(NativeEpochCounts::new()),
            evaluation: Evaluation::NativeAllocation,
        })
    }
    fn owner(&self, bytes: u64) -> Result<u64> {
        let mut l = NativeLedger::new(self.page, self.backend)?;
        l.bytes(bytes)?;
        Ok(l.total)
    }
    fn tensor(&self, elements: u64) -> Result<u64> {
        self.owner(mul(4, elements)?)
    }
    fn storage(
        &self,
        inventory: &Inventory,
        predicate: impl Fn(&str) -> bool,
        cast_f32: bool,
    ) -> Result<u64> {
        if self.evaluation == Evaluation::EpochCountOnly {
            return Ok(0);
        }
        inventory
            .0
            .iter()
            .filter(|(k, _)| predicate(k))
            .try_fold(0, |total, (_, t)| {
                let mut bytes = self.owner(t.bytes)?;
                // DC-AE constructors explicitly cast each retained source to F32;
                // account source and target together rather than inventing a drop.
                if cast_f32 && t.dtype != Dtype::Float32 {
                    bytes = sum(bytes, self.tensor(t.elements)?)?;
                }
                sum(total, bytes)
            })
    }
    fn coherence(&self, inventory: &Inventory, predicate: impl Fn(&str) -> bool) -> Result<u64> {
        if self.evaluation == Evaluation::EpochCountOnly {
            return Ok(0);
        }
        // A CPU compute stream can coexist with explicit default-GPU load
        // checks. Do not infer load-check absence from the compute backend.
        inventory
            .0
            .iter()
            .filter(|(k, t)| predicate(k) && t.bytes > 0)
            .try_fold(0, |total, (_, t)| {
                let words = if t.bytes.is_multiple_of(4) {
                    t.bytes / 4
                } else {
                    t.bytes
                };
                let mut cpu = NativeLedger::new(self.page, NativeBackend::Cpu)?;
                if !t.bytes.is_multiple_of(4) {
                    cpu.array(4, words)?;
                }
                cpu.reduce(4, words, 1, words, true)?;
                let mut gpu = NativeLedger::new(self.page, NativeBackend::Metal)?;
                for _ in 0..=mlx_gen::coherence::MAX_RETRIES {
                    if !t.bytes.is_multiple_of(4) {
                        gpu.array(4, words)?;
                    }
                    gpu.reduce(4, words, 1, words, true)?;
                }
                sum(total, sum(cpu.total, gpu.total)?)
            })
    }
    fn note(&self, label: &str) -> Result<()> {
        self.counts.borrow_mut().note(label)
    }
    fn quantum(
        &self,
        label: &'static str,
        ambient: u64,
        held: u64,
        build: impl FnOnce(&mut Walk<'a>) -> Result<()>,
    ) -> Result<u64> {
        self.note(label)?;
        if self.evaluation == Evaluation::EpochCountOnly {
            return Ok(0);
        }
        let mut w = Walk::new(self.facts, self.page, self.backend)?;
        build(&mut w)?;
        let here = sum(sum(ambient, held)?, w.ledger.total)?;
        self.peak.set(self.peak.get().max(here));
        Ok(here)
    }
    fn keep(&self, bytes: u64) {
        self.peak.set(self.peak.get().max(bytes));
    }
    fn caption_carrier(&self, guided: bool) -> Result<u64> {
        let (l, m, h) = (
            self.facts.gemma_tokens,
            self.facts.caption_tokens,
            dim(self.facts.gemma.hidden_size)?,
        );
        // First production gather is still lazy during the second Gemma pass;
        // fixture slices retain the full hidden parent too. Cover parent PLUS
        // possible gathered target, selected index and selected mask, each U.
        let one = sum(
            sum(self.tensor(mul(l, h)?)?, self.tensor(mul(m, h)?)?)?,
            mul(2, self.tensor(m)?)?,
        )?;
        mul(one, if guided { 2 } else { 1 })
    }
    fn text_weights(&self) -> Result<u64> {
        if self.evaluation == Evaluation::EpochCountOnly {
            return Ok(0);
        }
        let mut total = self.storage(&self.facts.text_encoder, |_| true, false)?;
        // Header sources remain a valid upper after one-plus norm targets form.
        // The targets and typed scalar source/cast are source-distinct owners.
        for (name, t) in &self.facts.text_encoder.0 {
            if name.ends_with("layernorm.weight") || name == "model.norm.weight" {
                total = sum(total, self.tensor(t.elements)?)?;
                total = sum(total, mul(2, self.tensor(1)?)?)?;
            }
        }
        Ok(total)
    }
    fn trunk_weights(&self) -> Result<u64> {
        if self.evaluation == Evaluation::EpochCountOnly {
            return Ok(0);
        }
        let inv = &self.facts.trunk;
        let mut total = self.storage(inv, |_| true, false)?;
        // Optional FusedQkvProjection packs can retain source and packed target.
        // Bound their source union, NOT a claim that every raw parent retired.
        for i in 0..self.facts.transformer.num_layers {
            for suffix in ["weight", "scales", "biases"] {
                let mut elements = 0;
                let mut dtype = None;
                let mut mixed = false;
                let mut count = 0;
                for part in ["to_q", "to_k", "to_v"] {
                    if let Some(t) = inv
                        .0
                        .get(&format!("transformer_blocks.{i}.attn1.{part}.{suffix}"))
                    {
                        elements = sum(elements, t.elements)?;
                        count += 1;
                        if dtype.is_some_and(|d| d != t.dtype) {
                            mixed = true;
                        }
                        dtype = Some(t.dtype);
                    }
                }
                if count == 3 {
                    let width = if mixed {
                        4
                    } else {
                        match dtype.unwrap() {
                            Dtype::Float16 | Dtype::Bfloat16 => 2,
                            _ => 4,
                        }
                    };
                    total = sum(total, self.owner(mul(width, elements)?)?)?;
                }
            }
        }
        Ok(total)
    }
    fn conditioning(&self, resident: u64, guided: bool) -> Result<u64> {
        let g = &self.facts.gemma;
        let l = self.facts.gemma_tokens;
        let h = dim(g.hidden_size)?;
        let hd = dim(g.head_dim)?;
        // Actual inputs, RoPE tables, mask, old/current hidden. Old hidden is
        // still held by forward's assignment while the guarded run completes.
        let fixed = sum(
            sum(mul(2, self.tensor(l)?)?, mul(2, self.tensor(mul(l, hd)?)?)?)?,
            self.tensor(mul(l, l)?)?,
        )?;
        let hidden = self.tensor(mul(l, h)?)?;
        let per_caption = self.caption_carrier(false)?;
        for prompt in 0..if guided { 2 } else { 1 } {
            let previous = if prompt == 0 { 0 } else { per_caption };
            self.quantum(GEMMA_LABELS[0], resident, previous, |w| {
                w.gemma_embedding(l, h)
            })?;
            for i in 0..g.num_layers {
                self.quantum(
                    GEMMA_LABELS[1],
                    resident,
                    sum(previous, sum(fixed, hidden)?)?,
                    |w| w.gemma_layer(i),
                )?;
            }
            self.quantum(
                GEMMA_LABELS[2],
                resident,
                sum(previous, sum(fixed, hidden)?)?,
                |w| w.rms(l, h, false),
            )?;
        }
        // Actual selected gather materialization (Sequential) or lazy carrier
        // (Resident) is retained conservatively; never discard its full parent.
        self.note(labels::TEXT_SELECTED)?;
        let captions = self.caption_carrier(guided)?;
        self.keep(sum(resident, captions)?);
        Ok(captions)
    }
    fn forward(
        &self,
        ops: &LiveOps,
        input: &LiveLatent,
        pending_native: u64,
    ) -> Result<LiveLatent> {
        let t = &self.facts.transformer;
        let p = dim(t.patch_size)?;
        let rows = mul(
            u64::from(self.plan.height / 32) / p,
            u64::from(self.plan.width / 32) / p,
        )?;
        let d = dim(t.inner_dim())?;
        let m = self.facts.caption_tokens;
        // The projected caption, temb, emb remain in forward across every
        // block/output. The input itself is already a real solver Rc owner.
        let persistent = sum(
            sum(self.tensor(mul(m, d)?)?, self.tensor(mul(6, d)?)?)?,
            self.tensor(d)?,
        )?;
        let hidden = self.tensor(mul(rows, d)?)?;
        let ambient = sum(ops.active_upper()?, ops.ambient())?;
        self.quantum(labels::ENTRY, ambient, pending_native, |w| {
            w.trunk_entry(self.plan)
        })?;
        // This detach models ONLY the source Entry eval+new joined fence.
        ops.materialize_after_join(input);
        let ambient = sum(ops.active_upper()?, ops.ambient())?;
        for i in 0..t.num_layers {
            self.quantum(labels::BLOCK, ambient, sum(persistent, hidden)?, |w| {
                w.trunk_block(self.plan, i)
            })?;
        }
        self.quantum(labels::OUTPUT, ambient, sum(persistent, hidden)?, |w| {
            w.trunk_output(self.plan)
        })?;
        let output = ops.fresh_materialized(mul(
            mul(
                u64::from(self.plan.height / 32),
                u64::from(self.plan.width / 32),
            )?,
            dim(t.out_channels)?,
        )?)?;
        Ok(output)
    }
}

include!("native_retired_dcae.rs");
include!("native_retired_sampler.rs");

pub(super) fn retired_upper(
    facts: &SanaPricingFacts,
    plan: &SanaPricingPlan,
    sampling: &NativeSampling,
    page: u64,
    backend: NativeBackend,
    analysis_budget: u64,
) -> Result<u64> {
    retired_profile(facts, plan, sampling, page, backend, analysis_budget).map(|(upper, _)| upper)
}
pub(super) fn retired_profile(
    facts: &SanaPricingFacts,
    plan: &SanaPricingPlan,
    sampling: &NativeSampling,
    page: u64,
    backend: NativeBackend,
    analysis_budget: u64,
) -> Result<(u64, NativeEpochCounts)> {
    let profile = RetiredProfile::new(facts, plan, page, backend)?;
    let upper = profile.request_peak(sampling, analysis_budget)?;
    let counts = *profile.counts.borrow();
    Ok((upper, counts))
}

/// SAME closed programme, epoch-count interpretation only. Its result type
/// contains no native bytes/certificate: skipped tensor-header/workspace
/// evaluation can never be mistaken for a zero native allocation bound.
pub(super) fn source_epoch_counts(
    facts: &SanaPricingFacts,
    plan: &SanaPricingPlan,
    sampling: &NativeSampling,
    page: u64,
    analysis_budget: u64,
) -> Result<NativeEpochCounts> {
    let mut profile = RetiredProfile::new(facts, plan, page, NativeBackend::Cpu)?;
    profile.evaluation = Evaluation::EpochCountOnly;
    let _non_authoritative = profile.request_peak(sampling, analysis_budget)?;
    let counts = *profile.counts.borrow();
    Ok(counts)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn conditional_epoch_peak_covers_closed_source_regimes_without_cumulative_step_growth() {
        for sprint in [false, true] {
            for reference in [false, true] {
                for tiled in [false, true] {
                    for bits in [0, 4, 8] {
                        let f = if bits == 0 {
                            super::super::tests::facts(sprint)
                        } else {
                            super::super::tests::packed(super::super::tests::facts(sprint), bits)
                        };
                        let p = super::super::tests::plan(sprint, reference, tiled);
                        let s = super::super::tests::sampling(sprint, reference, 5);
                        for backend in [NativeBackend::Cpu, NativeBackend::Metal] {
                            let profile = RetiredProfile::new(&f, &p, 16_384, backend).unwrap();
                            let peak = profile.request_peak(&s, 1 << 20).unwrap();
                            let cumulative = cumulative_upper(&f, &p, &s, 16_384, backend).unwrap();
                            assert!(peak>0&&peak<cumulative,"sprint={sprint} ref={reference} tiled={tiled} bits={bits} backend={backend:?}: {peak}/{cumulative}");
                            println!("conditional_retired_peak sprint={sprint} ref={reference} tiled={tiled} bits={bits} backend={backend:?} peak={peak} cumulative={cumulative}");
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn load_internal_sync_never_silently_erases_pending_check_owners() {
        let f = super::super::tests::facts(false);
        for reference in [false, true] {
            for optimized in [false, true] {
                let p = super::super::tests::plan(false, reference, optimized);
                let s = super::super::tests::sampling(false, reference, 5);
                let profile = RetiredProfile::new(&f, &p, 16_384, NativeBackend::Metal).unwrap();
                let reached_vae = profile
                    .coherence(&f.vae, |k| {
                        !p.sequential
                            || k.starts_with("decoder.")
                            || (p.img2img && k.starts_with("encoder."))
                    })
                    .unwrap();
                let pending = sum(
                    sum(
                        profile.coherence(&f.text_encoder, |_| true).unwrap(),
                        profile.coherence(&f.trunk, |_| true).unwrap(),
                    )
                    .unwrap(),
                    reached_vae,
                )
                .unwrap();
                assert!(pending > 0);
                let upper = profile.request_peak(&s, 1 << 20).unwrap();
                assert_eq!(
                    upper,
                    sum(profile.peak.get(), pending).unwrap(),
                    "no cross-quantum load-check retirement without complete source law"
                );
            }
        }
    }
    #[test]
    fn exact_source_epoch_counts_follow_solver_calls_serial_outputs_and_subquanta() {
        use mlx_gen::gen_core::sampling::Solver;
        for solver in Solver::ALL {
            for sprint in [false, true] {
                let f = super::super::tests::facts(sprint);
                let mut p = super::super::tests::plan(sprint, true, true);
                p.count = 3;
                let generation = mlx_gen::GenerationRequest {
                    steps: Some(5),
                    strength: Some(0.5),
                    sampler: (!sprint).then(|| solver.name().into()),
                    ..Default::default()
                };
                let sampling = super::super::super::native_certificate::fixture_sampling(
                    &generation,
                    sprint,
                    true,
                    1 << 20,
                )
                .unwrap();
                let profile = RetiredProfile::new(&f, &p, 16_384, NativeBackend::Metal).unwrap();
                profile.request_peak(&sampling, 1 << 20).unwrap();
                let counts = *profile.counts.borrow();
                let forwards = counts.count(labels::ENTRY).unwrap();
                assert!(forwards > 0);
                assert_eq!(
                    counts.count(labels::BLOCK).unwrap(),
                    forwards * f.transformer.num_layers as u64
                );
                assert_eq!(counts.count(labels::OUTPUT).unwrap(), forwards);
                assert_eq!(counts.count(labels::TEXT_SELECTED).unwrap(), 1);
                assert_eq!(
                    counts.count(labels::DECODED_PIXELS).unwrap(),
                    u64::from(p.count)
                );
                assert_eq!(
                    counts.count(GEMMA_LABELS[1]).unwrap(),
                    counts.count(GEMMA_LABELS[0]).unwrap() * f.gemma.num_layers as u64
                );
                assert_eq!(
                    counts.count(labels::DECODER_GLU).unwrap(),
                    counts.count(labels::DECODER_ATTENTION).unwrap()
                );
                assert_eq!(
                    counts.count(labels::ENCODER_GLU).unwrap(),
                    counts.count(labels::ENCODER_ATTENTION).unwrap()
                );
                if sprint {
                    assert_eq!(counts.count(labels::SCM_STEP).unwrap(), forwards);
                    assert_eq!(counts.count(labels::SCM_FINAL).unwrap(), u64::from(p.count));
                } else {
                    assert_eq!(
                        counts
                            .count(mlx_gen::sampler::SAMPLER_STEP_QUANTUM)
                            .unwrap(),
                        forwards / 2 + u64::from(p.count)
                    );
                }
            }
        }
    }
    #[test]
    fn event_only_interpretation_matches_native_program_without_becoming_a_certificate() {
        use mlx_gen::gen_core::sampling::Solver;
        for solver in Solver::ALL {
            for sprint in [false, true] {
                for reference in [false, true] {
                    for sequential in [false, true] {
                        for serial_count in [1, 3] {
                            let facts = super::super::tests::facts(sprint);
                            let mut plan = super::super::tests::plan(sprint, reference, true);
                            plan.sequential = sequential;
                            plan.count = serial_count;
                            let generation = mlx_gen::GenerationRequest {
                                steps: Some(3),
                                strength: Some(0.5),
                                sampler: (!sprint).then(|| solver.name().into()),
                                ..Default::default()
                            };
                            let sampling =
                                super::super::super::native_certificate::fixture_sampling(
                                    &generation,
                                    sprint,
                                    reference,
                                    1 << 20,
                                )
                                .unwrap();
                            let (_, actual) = retired_profile(
                                &facts,
                                &plan,
                                &sampling,
                                16_384,
                                NativeBackend::Metal,
                                1 << 20,
                            )
                            .unwrap();
                            let mut missing_headers = facts.clone();
                            missing_headers.text_encoder.0.clear();
                            missing_headers.trunk.0.clear();
                            missing_headers.vae.0.clear();
                            let planned = source_epoch_counts(
                                &missing_headers,
                                &plan,
                                &sampling,
                                16_384,
                                1 << 20,
                            )
                            .unwrap();
                            assert_eq!(planned, actual, "{solver:?} sprint={sprint} reference={reference} sequential={sequential} serial={serial_count}");
                            assert!(retired_upper(
                                &missing_headers, &plan, &sampling, 16_384, NativeBackend::Metal, 1 << 20,
                            ).is_err(), "count-only result must NOT authorize a zero-byte native certificate");
                        }
                    }
                }
            }
        }
    }
    #[test]
    fn ordinary_metadata_counts_include_solver_marks_and_only_one_shared_conditioning() {
        for (sprint, one_output, three_outputs) in [(false, 62, 168), (true, 38, 102)] {
            let facts = super::super::tests::facts(sprint);
            let generation = mlx_gen::GenerationRequest {
                steps: Some(3),
                sampler: (!sprint).then(|| "euler".into()),
                ..Default::default()
            };
            let sampling = super::super::super::native_certificate::fixture_sampling(
                &generation,
                sprint,
                false,
                1 << 20,
            )
            .unwrap();
            for (serial, expected) in [(1, one_output), (3, three_outputs)] {
                let mut plan = super::super::tests::plan(sprint, false, false);
                plan.count = serial;
                let counts =
                    source_epoch_counts(&facts, &plan, &sampling, 16_384, 1 << 20).unwrap();
                assert_eq!(
                    counts.solver_marks,
                    if sprint { 0 } else { 6 * u64::from(serial) }
                );
                let actual = super::super::super::native_certificate::ordinary_source_marks(
                    &counts, &plan, &sampling,
                )
                .unwrap();
                assert_eq!(
                    actual, expected,
                    "source metadata lost a real emission or repeated shared conditioning"
                );
            }
        }
    }
    #[test]
    fn source_histories_not_total_steps_determine_retired_solver_peak() {
        let f = super::super::tests::facts(false);
        let p = super::super::tests::plan(false, false, true);
        let a = super::super::tests::sampling(false, false, 5);
        let b = super::super::tests::sampling(false, false, 20);
        let small = RetiredProfile::new(&f, &p, 16_384, NativeBackend::Metal)
            .unwrap()
            .request_peak(&a, 1 << 20)
            .unwrap();
        let large = RetiredProfile::new(&f, &p, 16_384, NativeBackend::Metal)
            .unwrap()
            .request_peak(&b, 1 << 20)
            .unwrap();
        assert_eq!(
            small, large,
            "Euler has no growing history and real per-quantum retirement"
        );
    }
}
