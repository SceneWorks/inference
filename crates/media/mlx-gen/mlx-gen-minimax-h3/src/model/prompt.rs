//! Prompt encoding: the text encoder build and the plain, grounded and Ref2VA encodes.

use super::*;

impl MiniMaxH3 {
    /// Build the request's text encoder at the residency [`Self::requested_transformer_window`]
    /// admitted: resident over the mapped shards, or deferred with a per-window rebuild (sc-18662).
    ///
    /// One constructor for all three conditioning paths, so `t2va` cannot stream while `fl2va` /
    /// `ref2va` silently stay resident — the windowed walk lives in the encoder's own `run_layers`,
    /// which every forward variant routes through.
    pub(super) fn build_te(
        &self,
        w: Option<&Weights>,
        cfg: &MiniMaxH3TeConfig,
        window: Option<u32>,
        cancel: &mlx_gen::gen_core::CancelFlag,
    ) -> Result<MiniMaxH3TextEncoder> {
        match window {
            None => MiniMaxH3TextEncoder::from_weights(
                w.expect("a resident encoder builds from the mapped shards"),
                LM_PREFIX,
                cfg,
            ),
            Some(window) => {
                let mut te = MiniMaxH3TextEncoder::from_dir_deferred(
                    &self.text_encoder_dir,
                    LM_PREFIX,
                    cfg,
                )?;
                te.set_block_window(window as usize, cancel.clone())?;
                Ok(te)
            }
        }
    }

    /// Encode the prompt and immediately release the 66.7 GB text encoder.
    pub(super) fn encode_prompt(
        &self,
        prompt: &str,
        window: Option<u32>,
        cancel: &mlx_gen::gen_core::CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<mlx_rs::Array> {
        #[cfg(test)]
        if self.fixture.is_some() {
            return Ok(crate::veda_fixture::context());
        }
        let tok = MiniMaxH3Tokenizer::from_snapshot(&self.root)?;
        let (ids, mask) = tok.encode_prompt(prompt)?;

        // Under a deferred load the map is skipped entirely: `from_dir_deferred` reads only
        // `embed_tokens`, and a retained shard map would keep every layer tensor reachable —
        // the retained-view hazard `block_stream`'s docs name.
        let w = match window {
            None => Some(self.map_te_shards(false)?),
            Some(_) => None,
        };
        let cfg = MiniMaxH3TeConfig::qwen3_vl_32b();
        let te = self.build_te(w.as_ref(), &cfg, window, cancel)?;
        on_progress(Progress::Loading(
            mlx_gen::gen_core::LoadPhase::EncodingPrompt,
        ));
        let context = te.forward(&ids, &mask)?;
        // Force it BEFORE the encoder is dropped: under lazy evaluation the context is a graph node
        // holding every weight it was computed from, so dropping first would free nothing and the
        // first denoise step would re-materialize 66.7 GB.
        mlx_rs::transforms::eval([&context])?;
        release((te, w));
        Ok(context)
    }

    /// The `fl2va` presentation: run the Qwen3-VL **vision tower** over the keyframes, splice their
    /// embeddings into the `"<Picture i>: "` + vision-block presentation, and return the context
    /// together with its **per-row modality tags**.
    ///
    /// This is one of the two paths a keyframe takes. The other is the VAE encode in
    /// [`crate::conditioning`]; the sc-17242 spike established that the reference runs **both**,
    /// and neither substitutes for the other — the tower supplies semantic context to the prompt
    /// stream, the VAE supplies the pixel-space anchor the video rows are conditioned on.
    pub(super) fn encode_prompt_grounded(
        &self,
        prompt: &str,
        keyframes: &[&mlx_gen::media::Image],
        window: Option<u32>,
        cancel: &mlx_gen::gen_core::CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<(mlx_rs::Array, Vec<i32>)> {
        let tok = MiniMaxH3Tokenizer::from_snapshot(&self.root)?;
        let mut w = self.map_te_shards(true)?;

        let vision = VisionTower::from_weights(
            &w,
            crate::text_encoder::minimax_h3_vision_config(),
            VISION_PREFIX,
            VISION_GROUP_SIZE,
        )?;
        // sc-23402: the tower's 351 source tensors are the read set of `w` at this point — force
        // and GPU-verify them (sc-22414) before the first tower kernel touches them cold. Peak
        // -neutral: the tower is resident for the whole `run_vision` either way.
        w.materialize_accessed()?;
        let grounded = crate::text_encoder::run_vision(&vision, keyframes)?;
        // Force the tower's output BEFORE dropping it, and drop its tensors out of `w` too — the
        // same discipline `encode_prompt` documents. Under lazy evaluation `grounded` is a graph
        // node holding every weight it was computed from, so releasing the tower while `w` still
        // maps `model.visual.*` frees nothing and the splice below would re-materialize it.
        let mut forced: Vec<&mlx_rs::Array> = grounded.embeds.iter().collect();
        forced.extend(grounded.deepstack.iter().flatten());
        mlx_rs::transforms::eval(forced)?;
        release(vision);
        w.remove_prefix(VISION_PREFIX);

        let (ids, mask, tags) = tok.encode_fl2va(prompt, &grounded.counts)?;
        let cfg = MiniMaxH3TeConfig::qwen3_vl_32b();
        // Under a window the mapped shards are released before the encoder is built: the deferred
        // loader reopens the staged directory per window, and a retained shard map would keep every
        // layer tensor reachable across the whole windowed walk.
        let w = match window {
            None => Some(w),
            Some(_) => {
                release(w);
                None
            }
        };
        let te = self.build_te(w.as_ref(), &cfg, window, cancel)?;
        on_progress(Progress::Loading(
            mlx_gen::gen_core::LoadPhase::EncodingPrompt,
        ));
        let context = te.forward_with_images(
            &ids,
            &mask,
            &grounded.embeds,
            &grounded.deepstack,
            &grounded.grids,
        )?;
        mlx_rs::transforms::eval([&context])?;
        release((te, w, grounded));

        // The tags describe the presentation row for row; a mismatch would mis-tag every row after
        // the divergence and is silent, because both lengths build a runnable sequence.
        if context.shape()[1] != tags.len() as i32 {
            return Err(Error::Msg(format!(
                "{MODEL_ID}: the grounded context has {} rows but {} modality tags",
                context.shape()[1],
                tags.len()
            )));
        }
        Ok((context, tags))
    }
}

impl MiniMaxH3 {
    /// The `ref2va` presentation: the vision tower over every **visual** reference, spliced into
    /// the `<Picture i>` / `<Audio j>` / `<Video k>` presentation.
    ///
    /// A waveform never reaches the tower — a standalone audio reference contributes its label and
    /// nothing else, which is why the tower's sources and the presentation entries are built in one
    /// pass rather than zipped afterwards.
    pub(super) fn encode_prompt_ref2va(
        &self,
        prompt: &str,
        references: &[Ref2VaReference],
        window: Option<u32>,
        cancel: &mlx_gen::gen_core::CancelFlag,
        on_progress: &mut dyn FnMut(Progress),
    ) -> Result<(mlx_rs::Array, Vec<i32>)> {
        use crate::reference::{
            sample_video_condition_frames, ReferencePresentation, VIDEO_SAMPLE_FPS,
            VISION_TEMPORAL_PATCH,
        };

        let tok = MiniMaxH3Tokenizer::from_snapshot(&self.root)?;
        let mut w = self.map_te_shards(true)?;

        // The tower's sources, in **sequence order** — the order the pad runs appear in, which is
        // what `forward_with_references` consumes them in.
        let mut sources: Vec<crate::text_encoder::VisionSource<'_>> = Vec::new();
        // Per reference: how many tower sources it contributes, and its block timestamps.
        let mut plan: Vec<(usize, Vec<f64>)> = Vec::new();
        for r in references {
            match r {
                Ref2VaReference::Audio(_) => plan.push((0, Vec::new())),
                Ref2VaReference::Image(img) => {
                    sources.push(crate::text_encoder::VisionSource::Image(img));
                    plan.push((1, Vec::new()));
                }
                Ref2VaReference::Video(v) => {
                    let (indices, timestamps) = sample_video_condition_frames(
                        v.frames.len(),
                        crate::denoise::MINIMAX_H3_FPS,
                        VIDEO_SAMPLE_FPS,
                        VISION_TEMPORAL_PATCH,
                    )?;
                    // One tower input per merged group — the block the tokenizer presents — not one
                    // per sampled frame. A trailing partial group repeats its last frame, exactly
                    // as its timestamp label does.
                    let groups =
                        crate::reference::condition_frame_groups(&indices, VISION_TEMPORAL_PATCH);
                    if groups.len() != timestamps.len() {
                        return Err(Error::Msg(format!(
                            "{MODEL_ID}: a reference clip sampled {} frame group(s) for {} block \
                             timestamp(s)",
                            groups.len(),
                            timestamps.len()
                        )));
                    }
                    for &(first, second) in &groups {
                        sources.push(crate::text_encoder::VisionSource::FramePair(
                            &v.frames[first],
                            &v.frames[second],
                        ));
                    }
                    plan.push((groups.len(), timestamps));
                }
            }
        }
        if sources.is_empty() {
            return Err(Error::Msg(format!(
                "{MODEL_ID}: a ref2va request carries no visual reference for the conditioner — a \
                 waveform never reaches it"
            )));
        }

        let vision = VisionTower::from_weights(
            &w,
            crate::text_encoder::minimax_h3_vision_config(),
            VISION_PREFIX,
            VISION_GROUP_SIZE,
        )?;
        // sc-23402: same as `encode_prompt_grounded` — the tower is the FIRST big load of a
        // `ref2va` request and the packed token table the second; both are verified at their
        // load boundary now, the table in `MiniMaxH3TextEncoder::from_weights`.
        w.materialize_accessed()?;
        let grounded = crate::text_encoder::run_vision_sources(&vision, &sources)?;
        let mut forced: Vec<&mlx_rs::Array> = grounded.embeds.iter().collect();
        forced.extend(grounded.deepstack.iter().flatten());
        mlx_rs::transforms::eval(forced)?;
        release(vision);
        w.remove_prefix(VISION_PREFIX);

        // The presentation, from the tower's per-source counts.
        let mut cursor = 0usize;
        let mut presentation: Vec<ReferencePresentation> = Vec::with_capacity(references.len());
        for (r, (n_sources, timestamps)) in references.iter().zip(&plan) {
            match r {
                Ref2VaReference::Audio(_) => presentation.push(ReferencePresentation::Audio),
                Ref2VaReference::Image(_) => {
                    presentation.push(ReferencePresentation::Image {
                        num_tokens: grounded.counts[cursor],
                    });
                    cursor += 1;
                }
                Ref2VaReference::Video(v) => {
                    // One block per merged frame pair; every block of one clip shares a token count.
                    let num_tokens = grounded.counts[cursor];
                    presentation.push(ReferencePresentation::Video {
                        num_tokens,
                        timestamps: timestamps.clone(),
                        has_audio: v.audio.is_some(),
                    });
                    cursor += n_sources;
                }
            }
        }

        let (ids, mask, tags) = tok.encode_ref2va(prompt, &presentation)?;
        let cfg = MiniMaxH3TeConfig::qwen3_vl_32b();
        // Same release-before-build as `encode_prompt_grounded`: a retained shard map would defeat
        // the windowed walk's release.
        let w = match window {
            None => Some(w),
            Some(_) => {
                release(w);
                None
            }
        };
        let te = self.build_te(w.as_ref(), &cfg, window, cancel)?;
        on_progress(Progress::Loading(
            mlx_gen::gen_core::LoadPhase::EncodingPrompt,
        ));
        let context = te.forward_with_references(
            &ids,
            &mask,
            &grounded.embeds,
            &grounded.deepstack,
            &grounded.grids,
        )?;
        mlx_rs::transforms::eval([&context])?;
        release((te, w, grounded));

        if context.shape()[1] != tags.len() as i32 {
            return Err(Error::Msg(format!(
                "{MODEL_ID}: the ref2va context has {} rows but {} modality tags",
                context.shape()[1],
                tags.len()
            )));
        }
        Ok((context, tags))
    }
}
