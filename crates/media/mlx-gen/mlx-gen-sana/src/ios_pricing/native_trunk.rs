// Linear-DiT closed graph, included by native_graph.
impl Walk<'_> {
    fn fused_self_projection(&mut self, prefix: &str, rows: u64, input: u64) -> Result<()> {
        let inv = &self.facts.trunk;
        let mut outputs = 0;
        let mut codes = 0;
        let mut scales = 0;
        let mut group = None;
        let mut packed = true;
        for name in ["to_q", "to_k", "to_v"] {
            let base = format!("{prefix}.{name}");
            let output = self.linear(inv, &base, rows, input)?;
            outputs = sum(outputs, output)?;
            let weight = tensor(inv, &format!("{base}.weight"))?;
            codes = sum(codes, weight.elements)?;
            if let Some(s) = inv.0.get(&format!("{base}.scales")) {
                let ss = shape(s, 2)?;
                let g = input / ss[1];
                if group.is_some_and(|old| old != g) {
                    packed = false;
                }
                group = Some(g);
                scales = sum(scales, s.elements)?;
            } else {
                packed = false;
            }
        }
        // FusedQkvProjection's compatible pack performs ONE wider GEMM/QMM.
        // Split-K is shape dependent: charge that possible lowering IN ADDITION
        // to the individual projections, not three narrow workspace estimates.
        self.a(codes)?;
        self.a(mul(rows, input)?)?;
        if packed {
            self.ops(scales, 2)?;
            self.ledger.qmm(
                4,
                rows,
                input,
                outputs,
                [
                    codes,
                    scales,
                    group.ok_or_else(|| invalid("missing QKV affine group"))?,
                ],
            )?;
        } else {
            self.a(mul(outputs, input)?)?;
            self.mm(1, rows, outputs, input)?;
        }
        Ok(())
    }
    fn trunk(&mut self, plan: &SanaPricingPlan) -> Result<()> {
        self.trunk_entry(plan)?;
        for i in 0..self.facts.transformer.num_layers {
            self.trunk_block(plan, i)?;
        }
        self.trunk_output(plan)
    }
    fn trunk_entry(&mut self, plan: &SanaPricingPlan) -> Result<()> {
        let cfg = &self.facts.transformer;
        let inv = &self.facts.trunk;
        let (lh, lw) = (u64::from(plan.height / 32), u64::from(plan.width / 32));
        let patch = dim(cfg.patch_size)?;
        let rows = mul(lh / patch, lw / patch)?;
        let d = dim(cfg.inner_dim())?;
        let caption = self.facts.caption_tokens;
        self.conv(
            inv,
            "patch_embed.proj",
            lh,
            lw,
            dim(cfg.in_channels)?,
            patch,
            0,
            1,
        )?;
        self.a(mul(rows, d)?)?;
        self.timestep()?;
        self.a(256)?;
        let ts = if cfg.guidance_embeds {
            "time_embed.timestep_embedder"
        } else {
            "time_embed.emb.timestep_embedder"
        };
        let tsdim = self.linear(inv, &format!("{ts}.linear_1"), 1, 256)?;
        self.silu(tsdim)?;
        self.linear(inv, &format!("{ts}.linear_2"), 1, tsdim)?;
        if cfg.guidance_embeds {
            self.timestep()?;
            self.a(256)?;
            let g = self.linear(inv, "time_embed.guidance_embedder.linear_1", 1, 256)?;
            self.silu(g)?;
            self.linear(inv, "time_embed.guidance_embedder.linear_2", 1, g)?;
            self.a(d)?;
        }
        self.silu(d)?;
        self.linear(inv, "time_embed.linear", 1, d)?;
        let c = self.linear(
            inv,
            "caption_projection.linear_1",
            caption,
            dim(self.facts.gemma.hidden_size)?,
        )?;
        self.gelu(mul(caption, c)?)?;
        self.linear(inv, "caption_projection.linear_2", caption, c)?;
        self.rms(caption, d, false)?;
        Ok(())
    }
    fn trunk_block(&mut self, plan: &SanaPricingPlan, i: i32) -> Result<()> {
        let cfg = &self.facts.transformer;
        let inv = &self.facts.trunk;
        let (lh, lw) = (u64::from(plan.height / 32), u64::from(plan.width / 32));
        let patch = dim(cfg.patch_size)?;
        let rows = mul(lh / patch, lw / patch)?;
        let d = dim(cfg.inner_dim())?;
        let caption = self.facts.caption_tokens;
        let prefix = format!("transformer_blocks.{i}");
        self.ops(mul(6, d)?, 2)?; // table cast + table/temb sum
        self.norm(rows, d)?;
        self.modulate(rows, d)?;
        self.fused_self_projection(&format!("{prefix}.attn1"), rows, d)?;
        if cfg.qk_norm {
            self.rms(rows, d, false)?;
            self.rms(rows, d, false)?;
        }
        let heads = dim(cfg.num_attention_heads)?;
        if !d.is_multiple_of(heads) {
            return Err(invalid("self-attention head extent mismatch"));
        }
        let hd = d / heads;
        self.relu(mul(rows, d)?)?;
        self.relu(mul(rows, d)?)?;
        self.ops(mul(rows, d)?, 3)?; // three explicit F32 casts
        self.mm(heads, hd, hd, rows)?;
        self.mm(heads, hd, rows, hd)?;
        self.ledger.reduce(4, mul(rows, d)?, d, rows, true)?;
        self.mm(heads, 1, rows, hd)?;
        self.scalars(1)?;
        self.a(mul(heads, rows)?)?;
        self.ops(mul(rows, d)?, 3)?; // divide, output reshape, cast
        self.linear(inv, &format!("{prefix}.attn1.to_out.0"), rows, d)?;
        self.ops(mul(rows, d)?, 2)?; // gated self residual
        self.linear(inv, &format!("{prefix}.attn2.to_q"), rows, d)?;
        self.linear(inv, &format!("{prefix}.attn2.to_k"), caption, d)?;
        self.linear(inv, &format!("{prefix}.attn2.to_v"), caption, d)?;
        let cross = tensor(inv, &format!("{prefix}.attn2.to_q.weight"))?.shape[0];
        if cfg.qk_norm {
            self.rms(rows, cross, false)?;
            self.rms(caption, cross, false)?;
        }
        self.a(mul(rows, cross)?)?;
        self.ops(mul(caption, cross)?, 2)?;
        self.scalars(3)?;
        self.ops(caption, 3)?; // scale + additive caption mask
        let ch = dim(cfg.num_cross_attention_heads)?;
        if !cross.is_multiple_of(ch) {
            return Err(invalid("cross-attention head extent mismatch"));
        }
        let block = u64::try_from(plan.attention.query_block(
            1,
            cfg.num_cross_attention_heads,
            i32::try_from(rows).map_err(|_| invalid("query extent out of range"))?,
            i32::try_from(caption).map_err(|_| invalid("caption extent out of range"))?,
        ))
        .map_err(|_| invalid("negative query block"))?;
        if block == 0 {
            return Err(invalid("empty query block"));
        }
        let mut start = 0;
        while start < rows {
            let q = block.min(rows - start);
            self.mm(ch, q, caption, cross / ch)?;
            let scores = mul(ch, mul(q, caption)?)?;
            self.ops(scores, 2)?;
            self.ledger.softmax(4, scores)?;
            self.mm(ch, q, cross / ch, caption)?;
            start = sum(start, q)?;
        }
        self.ops(mul(rows, cross)?, 3)?; // chunk concat, reshape, dtype cast
        self.linear(inv, &format!("{prefix}.attn2.to_out.0"), rows, cross)?;
        self.a(mul(rows, d)?)?;
        self.norm(rows, d)?;
        self.modulate(rows, d)?;
        self.glu(
            inv,
            &format!("{prefix}.ff"),
            lh / patch,
            lw / patch,
            d,
            false,
        )?;
        self.ops(mul(rows, d)?, 4)?; // two reshape consumers + gate/residual

        Ok(())
    }
    fn trunk_output(&mut self, plan: &SanaPricingPlan) -> Result<()> {
        let cfg = &self.facts.transformer;
        let inv = &self.facts.trunk;
        let (lh, lw) = (u64::from(plan.height / 32), u64::from(plan.width / 32));
        let patch = dim(cfg.patch_size)?;
        let rows = mul(lh / patch, lw / patch)?;
        let d = dim(cfg.inner_dim())?;
        let _caption = self.facts.caption_tokens;
        self.ops(mul(2, d)?, 2)?;
        self.norm(rows, d)?;
        self.modulate(rows, d)?;
        let output = self.linear(inv, "proj_out", rows, d)?;
        self.ops(mul(rows, output)?, 2)
    }
}
