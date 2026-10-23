// Included by retired source profile. Real generic factories; explicit SCM
// expression graph, with only actual source join/detach boundaries simulated.
impl RetiredProfile<'_> {
    fn flow_one(
        &self,
        ops: &LiveOps,
        flow: &mlx_gen::sampler::flow_sampling::ResolvedFlowSampling,
        active_reference: bool,
        reference: Option<&LiveLatent>,
        n: u64,
    ) -> Result<LiveLatent> {
        use mlx_gen::gen_core::sampling::{denoise, FlowModelSampling};
        let noise = ops.random(n)?;
        let (initial, held_noise) = if active_reference {
            let clean =
                reference.ok_or_else(|| invalid("active init lacks source reference owner"))?;
            let a = ops.multiply_always(clean)?;
            let b = ops.multiply_always(&noise)?;
            (ops.binary(&a, &b)?, Some(noise))
        } else {
            (noise, None)
        };
        let guided = flow.effective_guidance().is_some_and(|g| g > 1.0);
        let mut preview = Walk::new(self.facts, self.page, self.backend)?;
        preview.preview(n, false)?;
        let preview_upper = preview.ledger.total;
        let ms = FlowModelSampling::new(flow.convention());
        let mut callback = |x: &LiveLatent, sigma: f32| {
            // Actual sampler::step_eval BEFORE the model callback. Histories
            // outside x's evaluated DAG remain owned, not globally discarded.
            self.note(mlx_gen::sampler::SAMPLER_STEP_QUANTUM)?;
            ops.materialize_after_join(x);
            denoise(ops, &ms, x, sigma, |input, _t| {
                let _timestep = ops.scalar()?;
                let cond = self.forward(ops, input, preview_upper)?;
                if guided {
                    let uncond = self.forward(ops, input, 0)?;
                    let difference = ops.binary(&cond, &uncond)?;
                    let scaled = ops.multiply_always(&difference)?;
                    ops.binary(&uncond, &scaled).map_err(Into::into)
                } else {
                    Ok(cond)
                }
            })
        };
        let output = flow
            .solver()
            .boxed::<LiveOps>()
            .sample(ops, &ms, &mut callback, initial, flow.sigmas(), 0)
            .map_err(Error::from)?;
        self.note(mlx_gen::sampler::SAMPLER_STEP_QUANTUM)?;
        ops.materialize_after_join(&output); // Actual final sampler::step_eval.
        drop(held_noise);
        Ok(output)
    }
    fn scm_one(
        &self,
        ops: &LiveOps,
        scm: &crate::sampling::ResolvedSprintSampling,
        active_reference: bool,
        reference: Option<&LiveLatent>,
        n: u64,
    ) -> Result<LiveLatent> {
        let noise = ops.random(n)?; // borrowed, not moved: source keeps it alive.
        let mut latents = if active_reference {
            let clean = reference.ok_or_else(|| invalid("active SCM init lacks reference"))?;
            let x0 = ops.multiply_always(clean)?;
            let noise_sd = ops.multiply_always(&noise)?;
            let x = ops.multiply_always(&x0)?;
            let y = ops.multiply_always(&noise_sd)?;
            ops.binary(&x, &y)?
        } else {
            ops.multiply_always(&noise)?
        };
        let _guidance = ops.scalar()?;
        let mut denoised = latents.clone();
        let mut preview = Walk::new(self.facts, self.page, self.backend)?;
        preview.preview(n, true)?;
        let preview_upper = preview.ledger.total;
        for epoch in scm.epochs() {
            self.note(labels::SCM_STEP)?;
            ops.materialize_after_join(&latents); // Actual SCM_STEP.
            let divided = ops.multiply_always(&latents)?;
            let lat_in = ops.multiply_always(&divided)?;
            let _time = ops.scalar()?;
            let raw = self.forward(ops, &lat_in, preview_upper)?;
            // scm_step's exact always-building operations (NOT scale(1)
            // aliases): a*lat_in + b*raw; /input_scale; *sigma_data.
            let a = ops.multiply_always(&lat_in)?;
            let b = ops.multiply_always(&raw)?;
            let sum = ops.binary(&a, &b)?;
            let normalized = ops.multiply_always(&sum)?;
            let model_output = ops.multiply_always(&normalized)?;
            let left = ops.multiply_always(&latents)?;
            let right = ops.multiply_always(&model_output)?;
            let pred_x0 = ops.binary(&left, &right)?;
            let next = if epoch.renoise {
                let draw = ops.random(n)?;
                let scaled = ops.multiply_always(&draw)?;
                let x = ops.multiply_always(&pred_x0)?;
                let y = ops.multiply_always(&scaled)?;
                ops.binary(&x, &y)?
            } else {
                pred_x0.clone()
            };
            denoised = pred_x0;
            latents = next;
        }
        let output = ops.multiply_always(&denoised)?; // Source final division.
        self.note(labels::SCM_FINAL)?;
        ops.materialize_after_join(&output); // Actual SCM_FINAL, including zero transitions.
        drop((noise, latents, denoised));
        Ok(output)
    }
    fn request_peak(&self, sampling: &NativeSampling, analysis_budget: u64) -> Result<u64> {
        let te = self.text_weights()?;
        let trunk = self.trunk_weights()?;
        let decoder = self.storage(&self.facts.vae, |k| k.starts_with("decoder."), true)?;
        let encoder = self.storage(&self.facts.vae, |k| k.starts_with("encoder."), true)?;
        let heavy = sum(trunk, decoder)?;
        let resident = sum(te, sum(heavy, encoder)?)?;
        let te_check = self.coherence(&self.facts.text_encoder, |_| true)?;
        let trunk_check = self.coherence(&self.facts.trunk, |_| true)?;
        let vae_check = self.coherence(&self.facts.vae, |k| {
            !self.plan.sequential || k.starts_with("decoder.")
        })?;
        let encoder_check = if self.plan.sequential && self.plan.img2img {
            self.coherence(&self.facts.vae, |k| k.starts_with("encoder."))?
        } else {
            0
        };
        // Ordinary load-internal synchronization does NOT prove captured-owner
        // destruction. Conservatively retain ALL reached check backing for the
        // complete request; do not subtract it at the first high-level quantum
        // until an actual complete relevant-stream receipt law is established.
        let unretired_load_checks =
            sum(sum(te_check, trunk_check)?, sum(vae_check, encoder_check)?)?;
        let reference_active = sampling.reference_active(self.plan.img2img);
        // Eager warm pair loads everything before conditioning. Sequential
        // drops text after actual selected-output materialize+retire; no credit
        // is granted to load-internal syncs before that explicit boundary.
        self.keep(if self.plan.sequential {
            sum(te, te_check)?
        } else {
            sum(resident, sum(te_check, sum(trunk_check, vae_check)?)?)?
        });
        let captions = self.conditioning(
            if self.plan.sequential { te } else { resident },
            sampling.guided(),
        )?;
        let n = mul(
            mul(
                u64::from(self.plan.height / 32),
                u64::from(self.plan.width / 32),
            )?,
            dim(self.facts.transformer.in_channels)?,
        )?;
        if self.plan.img2img && self.plan.sequential {
            self.keep(sum(captions, sum(encoder, encoder_check)?)?);
            if reference_active {
                self.encode_epochs(encoder, captions)?;
            }
        } else if reference_active {
            self.encode_epochs(resident, captions)?;
        }
        let ops = LiveOps::new(self.page, self.backend, analysis_budget)?;
        let reference = if reference_active {
            Some(ops.fresh_materialized(n)?)
        } else {
            None
        };
        let reference_upper = if reference_active { self.tensor(n)? } else { 0 };
        if self.plan.sequential {
            self.keep(sum(
                sum(captions, reference_upper)?,
                sum(heavy, sum(trunk_check, vae_check)?)?,
            )?);
        }
        ops.set_ambient(sum(
            if self.plan.sequential {
                heavy
            } else {
                resident
            },
            captions,
        )?)?;
        let mut outputs = Vec::new();
        for _ in 0..self.plan.count {
            let output = match sampling {
                NativeSampling::Flow { resolved, .. } => {
                    self.flow_one(&ops, resolved, reference_active, reference.as_ref(), n)?
                }
                NativeSampling::Scm(resolved) => {
                    self.scm_one(&ops, resolved, reference_active, reference.as_ref(), n)?
                }
            };
            outputs.push(output);
        }
        self.note(labels::LATENT_BATCH)?;
        // Real generator materializes all existing final owners, then drops
        // conditioning/reference and sheds trunk only for Sequential.
        drop(reference);
        let decode_weights = if self.plan.sequential {
            decoder
        } else {
            resident
        };
        ops.set_ambient(decode_weights)?;
        let all_latents = ops.active_upper()?;
        for _ in &outputs {
            self.decode_epochs(decode_weights, all_latents)?;
        }
        self.counts.borrow_mut().solver_marks = ops.solver_marks()?;
        self.keep(ops.peak());
        // Adding this source union to every candidate peak is intentionally
        // conservative (load-only candidates may already contain a subset).
        sum(self.peak.get(), unretired_load_checks)
    }
}
