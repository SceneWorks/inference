// Included by the private retired source profile; real source quanta and held
// image/head/halo/fold owners, never a measured-grid or logical scratch proof.
impl RetiredProfile<'_> {
    // Selected only after the matching production scope actually
    // splits/guards/joins these two subquanta. Old single-graph path CANNOT use it.
    fn dcae_block_epochs(
        &self,
        ambient: u64,
        held: u64,
        encode: bool,
        block: (usize, i32),
        h: u64,
        w: u64,
    ) -> Result<()> {
        let (i, j) = block;
        let cfg = &self.facts.dc_ae;
        let c = dim(cfg.block_out_channels[i])?;
        let input = self.tensor(mul(mul(h, w)?, c)?)?;
        if cfg.block_types[i] == BlockType::EfficientVit {
            let prefix = if encode {
                format!("encoder.down_blocks.{i}.{j}")
            } else {
                format!(
                    "decoder.up_blocks.{i}.{}",
                    j + i32::from(i + 1 < cfg.num_stages())
                )
            };
            self.quantum(
                if encode {
                    labels::ENCODER_ATTENTION
                } else {
                    labels::DECODER_ATTENTION
                },
                ambient,
                sum(held, input)?,
                |q| q.dcae_attn(&format!("{prefix}.attn"), h, w, c),
            )?;
            // Stage's old feature remains held while the attended result is
            // fed to GLU; retirement is not permission to erase either owner.
            self.quantum(
                if encode {
                    labels::ENCODER_GLU
                } else {
                    labels::DECODER_GLU
                },
                ambient,
                sum(held, mul(2, input)?)?,
                |q| {
                    q.glu(
                        &self.facts.vae,
                        &format!("{prefix}.conv_out"),
                        h,
                        w,
                        c,
                        true,
                    )
                },
            )?;
        } else {
            self.quantum(
                if encode {
                    labels::ENCODER_BLOCK
                } else {
                    labels::DECODER_BLOCK
                },
                ambient,
                sum(held, input)?,
                |q| q.dcae_block(encode, i, j, h, w),
            )?;
        }
        Ok(())
    }
    fn encode_stage_epochs(
        &self,
        ambient: u64,
        held: u64,
        range: std::ops::Range<usize>,
        mut h: u64,
        mut w: u64,
    ) -> Result<(u64, u64)> {
        let cfg = &self.facts.dc_ae;
        for i in range {
            let c = dim(cfg.block_out_channels[i])?;
            for j in 0..cfg.encoder_layers_per_block[i] {
                self.dcae_block_epochs(ambient, held, true, (i, j), h, w)?;
            }
            if i + 1 < cfg.num_stages() {
                let input = mul(mul(h, w)?, c)?;
                let next = dim(cfg.block_out_channels[i + 1])?;
                let (oh, ow) = (h.div_ceil(2), w.div_ceil(2));
                let output = mul(mul(oh, ow)?, next)?;
                self.quantum(
                    labels::ENCODER_DOWNSAMPLE,
                    ambient,
                    sum(held, self.tensor(input)?)?,
                    |q| {
                        q.conv(
                            &self.facts.vae,
                            &format!(
                                "encoder.down_blocks.{i}.{}.conv",
                                cfg.encoder_layers_per_block[i]
                            ),
                            h,
                            w,
                            c,
                            2,
                            1,
                            1,
                        )?;
                        q.a(input)?; // pixel-unshuffle reshape materialization
                        if output == 0 || !input.is_multiple_of(output) {
                            return Err(invalid("encoder shortcut extent mismatch"));
                        }
                        q.ledger.reduce(4, input, output, input / output, true)?;
                        q.scalars(1)?;
                        q.ops(output, 2)
                    },
                )?;
                (h, w) = (oh, ow);
            }
        }
        Ok((h, w))
    }
    fn encode_input_epoch(&self, ambient: u64, held: u64, h: u64, w: u64) -> Result<()> {
        self.quantum(labels::ENCODER_INPUT, ambient, held, |q| {
            q.conv(
                &self.facts.vae,
                "encoder.conv_in",
                h,
                w,
                dim(self.facts.dc_ae.in_channels)?,
                1,
                1,
                1,
            )?;
            Ok(())
        })?;
        Ok(())
    }
    fn encode_epochs(&self, ambient: u64, outer: u64) -> Result<()> {
        let cfg = &self.facts.dc_ae;
        let (h, w) = (u64::from(self.plan.height), u64::from(self.plan.width));
        let image = self.tensor(mul(mul(h, w)?, dim(cfg.in_channels)?)?)?;
        let base = sum(outer, image)?;
        // Host preprocessing uploads one F32 image, whose owner persists for
        // the whole encoder call (the NHWC is a view of that same owner).
        self.keep(sum(ambient, base)?);
        let prefix = cfg
            .block_types
            .iter()
            .take_while(|t| **t == BlockType::Res)
            .count();
        let downs = (0..prefix).filter(|i| i + 1 < cfg.num_stages()).count();
        let cell = 1u64
            .checked_shl(downs as u32)
            .ok_or_else(|| invalid("prefix scale overflow"))?;
        let core = self
            .plan
            .encode_tile
            .map(|v| dim(v).and_then(|v| mul(v.div_ceil(cell), cell)))
            .transpose()?;
        let long_parent;
        let (ph, pw) = if let Some(core) = core.filter(|v| prefix > 0 && (h > *v || w > *v)) {
            let (mut radius, mut scale) = (1u64, 1u64);
            for i in 0..prefix {
                radius = sum(
                    radius,
                    mul(mul(2, cfg.encoder_layers_per_block[i] as u64)?, scale)?,
                )?;
                if i + 1 < cfg.num_stages() {
                    radius = sum(radius, scale)?;
                    scale = mul(scale, 2)?;
                }
            }
            let halo = mul(radius.div_ceil(cell), cell)?;
            let extent = |len: u64| -> Result<Vec<(u64, u64)>> {
                let mut out = Vec::new();
                let mut start = 0;
                while start < len {
                    let end = sum(start, core)?.min(len);
                    out.push((
                        end - start,
                        sum(end, halo)?.min(len) - start.saturating_sub(halo),
                    ));
                    start = sum(start, core)?;
                }
                Ok(out)
            };
            let hs = extent(h)?;
            let ws = extent(w)?;
            let c = dim(cfg.block_out_channels[prefix.min(cfg.num_stages() - 1)])?;
            let mut parents = 0;
            for &(_, th) in &hs {
                for &(_, tw) in &ws {
                    parents = sum(parents, self.tensor(mul(mul(th / cell, tw / cell)?, c)?)?)?;
                }
            }
            // Every core crop may retain its entire halo feature parent. This
            // is per-source-parent U, NOT U(sum of logical cropped payloads).
            for &(_, th) in &hs {
                for &(_, tw) in &ws {
                    self.encode_input_epoch(ambient, sum(base, parents)?, th, tw)?;
                    self.encode_stage_epochs(ambient, sum(base, parents)?, 0..prefix, th, tw)?;
                }
            }
            let mut row_targets = 0;
            for &(core_h, _) in &hs {
                row_targets = sum(
                    row_targets,
                    self.tensor(mul(mul(core_h / cell, w / cell)?, c)?)?,
                )?;
            }
            let stitched = self.tensor(mul(mul(h / cell, w / cell)?, c)?)?;
            self.quantum(labels::ENCODER_STITCH, ambient, sum(base, parents)?, |q| {
                q.ledger.total = sum(q.ledger.total, sum(row_targets, stitched)?)?;
                Ok(())
            })?;
            long_parent = stitched;
            // Source drops rows immediately after the stitched quantum. The
            // whole stitched prefix remains owned through encode_rest.
            self.encode_stage_epochs(
                ambient,
                sum(base, stitched)?,
                prefix..cfg.num_stages(),
                h / cell,
                w / cell,
            )?
        } else {
            self.encode_input_epoch(ambient, base, h, w)?;
            let (ph, pw) = self.encode_stage_epochs(ambient, base, 0..prefix, h, w)?;
            let c = dim(cfg.block_out_channels[prefix.min(cfg.num_stages() - 1)])?;
            let prefix_owner = self.tensor(mul(mul(ph, pw)?, c)?)?;
            long_parent = prefix_owner;
            self.encode_stage_epochs(
                ambient,
                sum(base, prefix_owner)?,
                prefix..cfg.num_stages(),
                ph,
                pw,
            )?
        };
        let c = dim(*cfg
            .block_out_channels
            .last()
            .ok_or_else(|| invalid("empty encoder"))?)?;
        let input = mul(mul(ph, pw)?, c)?;
        let output = mul(mul(ph, pw)?, dim(cfg.latent_channels)?)?;
        self.quantum(
            labels::ENCODER_OUTPUT,
            ambient,
            sum(sum(base, long_parent)?, self.tensor(input)?)?,
            |q| {
                q.conv(&self.facts.vae, "encoder.conv_out", ph, pw, c, 1, 1, 1)?;
                if output == 0 || !input.is_multiple_of(output) {
                    return Err(invalid("encoder final shortcut mismatch"));
                }
                q.ledger.reduce(4, input, output, input / output, true)?;
                q.scalars(1)?;
                q.ops(output, 2)
            },
        )?;
        // Scaling is a separate real prepare_reference_latent materialization.
        self.quantum(
            labels::REFERENCE_LATENT,
            ambient,
            sum(base, self.tensor(output)?)?,
            |q| {
                q.scalars(1)?;
                q.a(output)
            },
        )?;
        Ok(())
    }
    fn decode_stage_epochs(
        &self,
        ambient: u64,
        held: u64,
        range: std::ops::Range<usize>,
        mut h: u64,
        mut w: u64,
    ) -> Result<(u64, u64)> {
        let cfg = &self.facts.dc_ae;
        for i in range.rev() {
            if i + 1 < cfg.num_stages() {
                let cin = dim(cfg.block_out_channels[i + 1])?;
                let cout = dim(cfg.block_out_channels[i])?;
                let input = self.tensor(mul(mul(h, w)?, cin)?)?;
                (h, w) = (mul(h, 2)?, mul(w, 2)?);
                self.quantum(labels::DECODER_UPSAMPLE, ambient, sum(held, input)?, |q| {
                    q.a(mul(mul(h, w)?, cin)?)?;
                    q.conv(
                        &self.facts.vae,
                        &format!("decoder.up_blocks.{i}.0.conv"),
                        h,
                        w,
                        cin,
                        1,
                        1,
                        1,
                    )?;
                    q.ops(mul(mul(h, w)?, cout)?, 3)
                })?;
            }
            for j in 0..cfg.layers_per_block[i] {
                self.dcae_block_epochs(ambient, held, false, (i, j), h, w)?;
            }
        }
        Ok((h, w))
    }
    fn decode_output_epoch(&self, ambient: u64, held: u64, h: u64, w: u64) -> Result<()> {
        let feature = self.tensor(mul(
            mul(h, w)?,
            dim(self.facts.dc_ae.block_out_channels[0])?,
        )?)?;
        self.quantum(labels::DECODER_OUTPUT, ambient, sum(held, feature)?, |q| {
            q.decode_output(h, w)
        })?;
        Ok(())
    }
    fn decode_epochs(&self, ambient: u64, outer: u64) -> Result<()> {
        let cfg = &self.facts.dc_ae;
        let (lh, lw) = (
            u64::from(self.plan.height / 32),
            u64::from(self.plan.width / 32),
        );
        let n = mul(mul(lh, lw)?, dim(cfg.latent_channels)?)?;
        // Both `scale` and `unscaled` are outer Rust owners until conversion
        // returns. Existing original latents are already in outer/solver Vec.
        let base = sum(outer, sum(self.tensor(n)?, self.tensor(1)?)?)?;
        self.quantum(labels::DECODER_INPUT, ambient, base, |q| {
            q.conv(
                &self.facts.vae,
                "decoder.conv_in",
                lh,
                lw,
                dim(cfg.latent_channels)?,
                1,
                1,
                1,
            )?;
            q.ops(
                mul(mul(lh, lw)?, dim(*cfg.block_out_channels.last().unwrap())?)?,
                2,
            )
        })?;
        let tail = cfg
            .block_types
            .iter()
            .take_while(|t| **t == BlockType::Res)
            .count();
        let (hh, hw) = self.decode_stage_epochs(ambient, base, tail..cfg.num_stages(), lh, lw)?;
        let head_c = dim(cfg.block_out_channels[tail.min(cfg.num_stages() - 1)])?;
        let head = self.tensor(mul(mul(hh, hw)?, head_c)?)?;
        let held = sum(base, head)?;
        let scale = 1u64
            .checked_shl((0..tail).filter(|i| i + 1 < cfg.num_stages()).count() as u32)
            .ok_or_else(|| invalid("tail scale overflow"))?;
        let vae = VaeTiling {
            spatial_scale: i32::try_from(scale).map_err(|_| invalid("tail scale exceeds axis"))?,
            temporal_scale: 1,
            causal_temporal: false,
            full_res_channels: 3,
        };
        let tiling = self
            .plan
            .decode_tiling
            .map(|(e, o)| TilingConfig::spatial_only(e, o));
        if let Some(tiling) =
            tiling.filter(|t| tail > 0 && t.needs_tiling(vae, 1, hh as i32, hw as i32))
        {
            let plan = tiling.plan(vae, 1, hh as i32, hw as i32);
            let (oh, ow) = (mul(hh, scale)?, mul(hw, scale)?);
            for ht in &plan.h {
                for wt in &plan.w {
                    let th =
                        u64::try_from(ht.end - ht.start).map_err(|_| invalid("negative tile"))?;
                    let tw =
                        u64::try_from(wt.end - wt.start).map_err(|_| invalid("negative tile"))?;
                    let h = mul(th, scale)?;
                    let w = mul(tw, scale)?;
                    // At most one source owner for each fold axis. Retain the
                    // current strip + slab + temporal output throughout this tile.
                    let acc = sum(
                        self.tensor(mul(mul(h, ow)?, 3)?)?,
                        mul(2, self.tensor(mul(mul(oh, ow)?, 3)?)?)?,
                    )?;
                    let tile_held = sum(held, acc)?;
                    self.decode_stage_epochs(ambient, tile_held, 0..tail, th, tw)?;
                    self.decode_output_epoch(ambient, tile_held, h, w)?;
                    self.quantum(
                        labels::TILE_FOLD,
                        ambient,
                        sum(tile_held, self.tensor(mul(mul(h, w)?, 3)?)?)?,
                        |q| {
                            q.scalars(4)?;
                            q.a(w)?;
                            q.ops(h, 2)?;
                            q.a(mul(h, w)?)?;
                            q.a(mul(mul(h, w)?, 3)?)?;
                            q.ops(mul(mul(h, ow)?, 3)?, 2)?;
                            q.ops(mul(mul(oh, ow)?, 3)?, 4)
                        },
                    )?;
                }
            }
            self.quantum(
                labels::TILE_FINALIZE,
                ambient,
                sum(held, self.tensor(mul(mul(oh, ow)?, 3)?)?)?,
                |q| {
                    q.scalars(2)?;
                    q.ops(oh, 2)?;
                    q.a(ow)?;
                    q.ops(mul(oh, ow)?, 2)?;
                    q.ops(mul(mul(oh, ow)?, 3)?, 2)
                },
            )?;
        } else {
            let (h, w) = self.decode_stage_epochs(ambient, held, 0..tail, hh, hw)?;
            self.decode_output_epoch(ambient, held, h, w)?;
        }
        let rgb = mul(
            mul(u64::from(self.plan.height), u64::from(self.plan.width))?,
            3,
        )?;
        // Shared conversion's input and its transpose are one real source
        // backing. All seven possible native outputs are separately charged.
        self.quantum(
            labels::DECODED_PIXELS,
            ambient,
            sum(base, self.tensor(rgb)?)?,
            |q| {
                q.scalars(4)?;
                q.ops(rgb, 7)
            },
        )?;
        Ok(())
    }
}
