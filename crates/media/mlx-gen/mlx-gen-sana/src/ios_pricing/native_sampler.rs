// Exact generic solver replay, included by native_graph. It deliberately uses
// the production Solver factory; no constant model-evaluation multiplier.
#[derive(Clone)]
struct NativeLatent(u64);
struct NativeOps(std::cell::RefCell<NativeLedger>);
impl mlx_gen::gen_core::sampling::LatentOps for NativeOps {
    type Latent = NativeLatent;
    fn scale(&self, x: &NativeLatent, c: f32) -> mlx_gen::gen_core::Result<NativeLatent> {
        if c != 1.0 {
            let mut l = self.0.borrow_mut();
            l.scalar(4)?;
            l.op(4, x.0)?;
        }
        Ok(x.clone())
    }
    fn add(&self, x: &NativeLatent, y: &NativeLatent) -> mlx_gen::gen_core::Result<NativeLatent> {
        if x.0 != y.0 {
            return Err(invalid("solver latent shape mismatch").into());
        }
        self.0.borrow_mut().op(4, x.0)?;
        Ok(x.clone())
    }
    fn sub(&self, x: &NativeLatent, y: &NativeLatent) -> mlx_gen::gen_core::Result<NativeLatent> {
        self.add(x, y)
    }
    fn axpy(
        &self,
        a: f32,
        x: &NativeLatent,
        _b: f32,
        y: &NativeLatent,
    ) -> mlx_gen::gen_core::Result<NativeLatent> {
        if x.0 != y.0 {
            return Err(invalid("solver latent shape mismatch").into());
        }
        let mut l = self.0.borrow_mut();
        l.scalar(4)?;
        l.op(4, y.0)?;
        if a != 1.0 {
            l.scalar(4)?;
            l.op(4, x.0)?;
        }
        l.op(4, x.0)?;
        Ok(x.clone())
    }
    fn randn_like(
        &self,
        x: &NativeLatent,
        _seed: u64,
        _step: usize,
    ) -> mlx_gen::gen_core::Result<NativeLatent> {
        let mut l = self.0.borrow_mut();
        l.array(4, 2)?;
        l.array(4, 4)?;
        l.repeated(4, 1, 6)?;
        l.repeated(4, x.0, 8)?;
        Ok(x.clone())
    }
}

pub(super) enum NativeSampling {
    Flow(mlx_gen::sampler::flow_sampling::ResolvedFlowSampling),
    Scm(crate::sampling::ResolvedSprintSampling),
}
impl NativeSampling {
    fn guided(&self) -> bool {
        matches!(self,Self::Flow(flow) if flow.effective_guidance().is_some_and(|g|g>1.0))
    }
}
impl Walk<'_> {
    fn preview(&mut self, n: u64, sprint: bool) -> Result<()> {
        if sprint {
            self.scalars(1)?;
            self.a(n)?;
        }
        // preview::project_latents: cast + potential reshape repair, factor and
        // bias uploads, matmul, bias, max0/min1, multiply255, round, readback copy.
        self.ops(n, 2)?;
        self.a(96)?;
        self.a(3)?;
        self.mm(1, n / 32, 3, 32)?;
        self.scalars(3)?;
        self.ops(mul(n / 32, 3)?, 6)
    }
    fn render(&mut self, plan: &SanaPricingPlan, sampling: &NativeSampling) -> Result<()> {
        let n = mul(
            mul(u64::from(plan.height / 32), u64::from(plan.width / 32))?,
            dim(self.facts.transformer.in_channels)?,
        )?;
        self.random(n)?;
        let mut forward = Walk::new(self.facts, self.ledger.page, self.ledger.backend)?;
        forward.trunk(plan)?;
        let forward_upper = forward.ledger.total;
        match sampling {
            NativeSampling::Flow(flow) => {
                if plan.img2img {
                    self.scalars(2)?;
                    self.ops(n, 3)?;
                }
                // Borrow no source/Array state into the boxed production factory.
                // Preserve the accumulated prefix even when symbolic replay errors.
                let empty = NativeLedger::new(self.ledger.page, self.ledger.backend)?;
                let owned = std::mem::replace(&mut self.ledger, empty);
                let ops = NativeOps(std::cell::RefCell::new(owned));
                let ms = mlx_gen::gen_core::sampling::FlowModelSampling::new(flow.convention());
                let guided = sampling.guided();
                let preview_upper = {
                    let mut p = Walk::new(self.facts, self.ledger.page, self.ledger.backend)?;
                    p.preview(n, false)?;
                    p.ledger.total
                };
                let mut predict =
                    |input: &NativeLatent, _t: f32| -> mlx_gen::gen_core::Result<NativeLatent> {
                        let mut l = ops.0.borrow_mut();
                        // Charge optional preview on EVERY callback (including correctors),
                        // an upper over the source's outer-step-only preview candidates.
                        l.total = sum(l.total, preview_upper)?;
                        l.scalar(4)?; // model timestep
                        l.total = sum(l.total, forward_upper)?;
                        if guided {
                            l.total = sum(l.total, forward_upper)?;
                            l.scalar(4)?;
                            l.repeated(4, n, 3)?; // true CFG subtract/mul/add
                        }
                        Ok(input.clone())
                    };
                let mut callback = |input: &NativeLatent, sigma: f32| {
                    mlx_gen::gen_core::sampling::denoise(&ops, &ms, input, sigma, &mut predict)
                };
                let result = flow
                    .solver()
                    .boxed::<NativeOps>()
                    .sample(&ops, &ms, &mut callback, NativeLatent(n), flow.sigmas(), 0)
                    .map_err(Error::from);
                self.ledger = ops.0.into_inner();
                result?;
            }
            NativeSampling::Scm(scm) => {
                self.scalars(1)?; // embedded guidance, once outside the SCM loop
                if plan.img2img {
                    // x0=clean*sd, noise_sd=noise*sd; two angle products and their sum.
                    self.scalars(4)?;
                    self.ops(n, 5)?;
                } else {
                    self.scalars(1)?;
                    self.a(n)?; // pure-noise sigma-data prior
                }
                for epoch in scm.epochs() {
                    self.preview(n, true)?;
                    self.scalars(3)?;
                    self.ops(n, 2)?; // lat/sd * input_scale; timestep
                    self.ledger.total = sum(self.ledger.total, forward_upper)?;
                    // scm_step model_output: a*lat_in, b*raw, sum, divide in_scale,
                    // multiply sigma_data, then two scaled branches and pred_x0.
                    self.scalars(6)?;
                    self.ops(n, 8)?;
                    if epoch.renoise {
                        self.random(n)?;
                        self.scalars(3)?;
                        self.ops(n, 4)?;
                    }
                }
                self.scalars(1)?;
                self.a(n)?; // terminal divide, also zero-transition path
            }
        }
        // model::materialize_latent_batch/concatenate and the per-image decode.
        self.a(n)?;
        self.decode(plan)
    }
}

/// A cumulative source allocation upper through complete load+generate. Every
/// source owner remains charged for the entire prefix: alias/drop/command-buffer
/// lifetime changes cannot create an unpriced retirement credit.
pub(super) fn cumulative_upper(
    facts: &SanaPricingFacts,
    plan: &SanaPricingPlan,
    sampling: &NativeSampling,
    page: u64,
    backend: NativeBackend,
) -> Result<u64> {
    if plan.count == 0
        || plan.width == 0
        || plan.height == 0
        || !plan.width.is_multiple_of(32)
        || !plan.height.is_multiple_of(32)
    {
        return Err(invalid("invalid serial count or DC-AE geometry"));
    }
    if plan.sprint != matches!(sampling, NativeSampling::Scm(_)) {
        return Err(invalid("sampler/variant mismatch"));
    }
    let d = &facts.dc_ae;
    let stages = d.num_stages();
    if stages == 0
        || d.block_types.len() != stages
        || d.layers_per_block.len() != stages
        || d.encoder_layers_per_block.len() != stages
        || d.block_out_channels.iter().any(|&n| n <= 0)
        || d.layers_per_block
            .iter()
            .chain(&d.encoder_layers_per_block)
            .any(|&n| n < 0)
    {
        return Err(invalid("incoherent DC-AE stage source"));
    }
    let spatial_scale = u32::try_from(stages - 1)
        .ok()
        .and_then(|n| 1u32.checked_shl(n));
    // These are create_noise's actual source contracts, not a private geometry
    // cap: an incompatible custom trunk/VAE cannot run this served generator.
    if spatial_scale != Some(crate::pipeline::SPATIAL_SCALE)
        || d.latent_channels != crate::pipeline::LATENT_CHANNELS
        || facts.transformer.in_channels != crate::pipeline::LATENT_CHANNELS
        || facts.transformer.out_channels != crate::pipeline::LATENT_CHANNELS
        || plan.width > i32::MAX as u32
        || plan.height > i32::MAX as u32
    {
        return Err(invalid(
            "source does not match the served f32c32 noise/latent geometry",
        ));
    }
    let mut prefix = Walk::new(facts, page, backend)?;
    prefix.inventory(&facts.text_encoder)?;
    prefix.gemma(sampling.guided())?;
    // Resident warm heavy includes the reference encoder even for text; staging
    // may load the encoder separately before decoder/trunk. Charging VAE twice
    // explicitly covers that reload, with no cancellation/drop retirement credit.
    if plan.img2img && plan.sequential {
        prefix.inventory(&facts.vae)?;
    }
    prefix.inventory(&facts.trunk)?;
    prefix.inventory(&facts.vae)?;
    if plan.img2img {
        prefix.encode(plan)?;
    }
    let mut render = Walk::new(facts, page, backend)?;
    render.render(plan, sampling)?;
    sum(
        prefix.ledger.total,
        mul(render.ledger.total, u64::from(plan.count))?,
    )
}
