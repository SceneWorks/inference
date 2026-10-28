//! [`LlamaProvider`](super::LlamaProvider)'s text-generation contract (sc-27302): the
//! `TextLlm` implementation — request validation and admission, prompt rendering, the decode
//! routes, streaming, tool-call and reasoning channels — and the request report it returns.

use super::*;

/// Run a piece of answer-channel text through the tool-call segmenter when active, returning the
/// plain-content runs to stream (tool-call blocks lifted out + parsed into [`ToolCallSegmenter`]).
/// With no segmenter the text passes straight through, so the non-tools path is byte-identical to
/// before.
pub(super) fn tool_pieces(seg: &mut Option<ToolCallSegmenter>, text: &str) -> Vec<String> {
    match seg {
        Some(ts) => ts.push(text),
        None => vec![text.to_string()],
    }
}

/// Emit one answer-channel content `piece` as a [`Channel::Content`] token event with the gap-free
/// `emit_index`, accumulating it into `streamed`. Shared by the streaming loop and the
/// end-of-generation tails; `*emit_index` / `*last_id` advance only when text is actually emitted, so
/// the contract's token index stays gap-free across stripped reasoning markers and lifted-out
/// tool-call blocks.
pub(super) fn emit_content(
    piece: &str,
    id: u32,
    stop: (&mut StopMatcher, &std::cell::Cell<bool>),
    streamed: &mut String,
    emit_index: &mut usize,
    last_id: &mut u32,
    on_event: &mut dyn FnMut(CoreEvent),
) {
    let (stop_matcher, halt) = stop;
    if halt.get() {
        return;
    }
    let chunk = stop_matcher.push(piece);
    if !chunk.emit.is_empty() {
        streamed.push_str(&chunk.emit);
        *last_id = id;
        on_event(CoreEvent::Token {
            id,
            text: chunk.emit,
            index: *emit_index,
            channel: Channel::Content,
        });
        *emit_index += 1;
    }
    if chunk.stop {
        halt.set(true);
    }
}

impl TextLlm for LlamaProvider {
    fn descriptor(&self) -> &TextLlmDescriptor {
        &self.descriptor
    }

    fn synchronize_before_release(&self) -> CoreResult<()> {
        // This is the decoder/VLM's OWN device, including an explicitly selected
        // non-blocking CUDA stream; an image generator's stream is not a fence for it.
        self.model
            .device()
            .synchronize()
            .map_err(|error| to_core(error.into()))
    }

    fn validate(&self, req: &TextLlmRequest) -> CoreResult<()> {
        self.descriptor
            .capabilities
            .validate_request(&self.descriptor.id, req)
    }

    fn load_report(&self) -> Option<core_llm::LoadReport> {
        Some(core_llm::LoadReport {
            draft: self.draft_report.clone(),
            ..self.load_record.report()
        })
    }

    fn generate(
        &self,
        req: &TextLlmRequest,
        on_event: &mut dyn FnMut(CoreEvent),
    ) -> CoreResult<TextLlmOutput> {
        // Request start: a request that fails or is cancelled (anywhere below, before the record
        // is written on success) must not leave the previous request's record readable.
        *self
            .last_decode
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = None;
        self.validate(req)?;
        if req.cancel.is_cancelled() {
            return Err(CoreError::Canceled); // typed pre-inference cancel
        }
        // The load's CUDA-graph policy (sc-24139) governs everything this request does on this
        // thread — admission's workspace pricing, wrapping the step model, the runner's switch —
        // and the report says which switch it ran under.
        // A provider built from parts (no load settled a policy) takes its model's device's
        // default — never the default of the device this process would select (sc-24446: a CPU
        // model in a CUDA build would otherwise price graph memory it can never capture).
        let policy = self.load_record.cuda_graphs.unwrap_or_else(|| {
            crate::decode::graph::cuda_graphs_default_for(self.model.device().is_cuda())
        });
        let _cuda_graphs_scope = crate::decode::cuda_graphs_scope(Some(policy));
        let cuda_graphs_on = cuda_graphs_enabled();

        // Multimodal (Qwen-VL + image/video content): replace image/video blocks with the Qwen-VL
        // placeholder text so the (vision-free) chat template renders the vision framing
        // (`<|vision_start|><|image_pad|><|vision_end|>` per image; the per-frame Text–Timestamp-
        // Alignment string per video). The visuals are encoded + spliced after tokenizing. Text-only
        // requests are unchanged.
        let temporal_patch = self
            .vision
            .as_ref()
            .map(|v| v.processor.temporal_patch_size)
            .unwrap_or(2);
        let (images, videos): (Vec<&ImageRef>, Vec<&VideoRef>) = match &self.vision {
            Some(_) => (collect_images(&req.messages), collect_videos(&req.messages)),
            None => (Vec::new(), Vec::new()),
        };
        let multimodal = !images.is_empty() || !videos.is_empty();
        // Gemma 4 multimodal (sc-18772): its own marker substitution and prefill, taken whenever
        // this is a Gemma 4 checkpoint with front-ends AND the request actually carries a visual or
        // a clip. `validate` has already rejected a modality this checkpoint does not ship.
        let gemma4_mm_request = self.gemma4.is_some()
            && (!collect_images(&req.messages).is_empty()
                || !collect_audio(&req.messages).is_empty());
        let substituted;
        let messages: &[Message] = if gemma4_mm_request {
            substituted = substitute_gemma4_placeholders(&req.messages)?;
            &substituted
        } else if multimodal {
            substituted = substitute_vision_placeholders(&req.messages, temporal_patch)?;
            &substituted
        } else {
            &req.messages
        };

        // Render the conversation and tokenize. The template already includes BOS, so encode without
        // auto special tokens. `enable_thinking` flows into the template kwarg so a no-think
        // (Disabled) request injects the model's closed `<think></think>` generation prompt; Auto
        // omits the kwarg (template default).
        let prompt = self.template.render_with(
            messages,
            &RenderOptions {
                add_generation_prompt: true,
                enable_thinking: req.enable_thinking_kwarg(),
                reasoning_effort: req.reasoning_effort,
                preserve_thinking: req.preserve_thinking,
                tools: &req.tools,
                date_string: None,
            },
        )?;
        let prompt_ids: Vec<i32> = self
            .tokenizer
            .encode(&prompt, false)?
            .into_iter()
            .map(|id| id as i32)
            .collect();

        // Price the request before image/video preprocessing or any model forward allocates native
        // tensors. The visual geometry path is pure integer math over request dimensions.
        let (visual_tokens, vision_workspace) = match &self.vision {
            Some(vision) if multimodal => vision.estimate_workspace(&images, &videos)?,
            _ => (0, 0),
        };
        let vision_workspace = vision_workspace
            .checked_add(request_media_workspace_bytes(&req.messages)?)
            .ok_or_else(|| CoreError::InvalidRequest("media workspace overflow".into()))?;
        let replaced_visual_placeholders =
            self.vision
                .as_ref()
                .filter(|_| multimodal)
                .map_or(0, |vision| {
                    prompt_ids
                        .iter()
                        .filter(|&&id| id == vision.image_token_id || id == vision.video_token_id)
                        .count()
                });
        let admitted_prompt = prompt_ids
            .len()
            .checked_sub(replaced_visual_placeholders)
            .and_then(|n| n.checked_add(visual_tokens))
            .ok_or_else(|| CoreError::InvalidRequest("expanded prompt geometry overflow".into()))?;
        // Pure prompt/visual geometry is known before native preprocessing. Architectural context
        // overflow must win over a transient capacity failure on the current host or CUDA device.
        validate_context_window(
            self.descriptor.capabilities.max_context_tokens,
            admitted_prompt,
            req.max_new_tokens,
        )?;
        // sc-20683: the request's KV cache under the shared compressed-KV policy — always dense
        // here, with the reason MLX's decision for the same request gives (or `ReaderUnavailable`
        // where MLX would compress).
        let kv_cache = self.kv_cache_plan(
            req.kv_compression,
            u64::try_from(admitted_prompt).unwrap_or(u64::MAX),
            req.max_new_tokens,
            1,
            multimodal || gemma4_mm_request,
        );
        // Every portable eager-attention mask variant is query-tiled; CUDA flash attention is
        // bounded more tightly. Use the runtime's exact maximum tile so admission prices the same
        // peak score/mask/weight lifetime the implementation enforces.
        // The backend-neutral speculative resolution (core-llm, sc-24433): `auto` runs MTP where
        // the checkpoint has a head, else prompt lookup; anything the plan cannot deliver is
        // named in the report's `fallbacks` (epic sc-24432 E2), never a silent downgrade.
        // A `draft_model` request reaching past the draft's own context window runs `auto`
        // instead, by name (sc-24436, E2). A request that leaves the option unset runs this
        // device's defaults-table default (E5, sc-24446).
        let speculative = req.speculative_or(self.speculative_default);
        // The engine gets the option itself: under `auto` it monitors the proposer it runs and
        // demotes one measurably slower than plain decoding to its ordinary steps (sc-24446, E5);
        // an explicit `{proposer, depth}` runs as asked. (Named apart from the MTP branch's
        // `speculative` flag, which shadows `speculative` below.)
        let requested_speculative = speculative;
        let resolution = core_llm::fit_draft_context(
            core_llm::resolve_speculative(speculative, &self.descriptor.capabilities),
            &self.descriptor.capabilities,
            self.draft_context(),
            admitted_prompt,
            req.max_new_tokens,
        );
        // Which loop decodes this request (sc-24140): decided once, so admission prices the
        // cache and graphs that loop will actually hold.
        let (mut route, route_fallback) = self.decode_route(resolution.plan);
        let mut fallbacks: Vec<String> = resolution
            .fallback
            .into_iter()
            .chain(route_fallback)
            .collect();
        let required = self
            .request_bytes(route, admitted_prompt, req.max_new_tokens, vision_workspace)
            .ok_or_else(|| CoreError::InvalidRequest("request memory estimate overflow".into()))?;
        // The cross-turn prefix cache (sc-24437) serves the engine's text routes. It keys on token
        // ids, which cannot tell two images (or clips) behind the same placeholder ids apart — a
        // multimodal prompt neither reads nor feeds it — and the reference loop is the parity
        // oracle, which always prefills cold; the report names either. The hybrid decoder
        // snapshots at the end of the rendered conversation, the prefix the next turn extends.
        let prefix_on = self.prefix_cache().budget_bytes() > 0;
        let prefix_route = !multimodal && !gemma4_mm_request && route != DecodeRoute::Reference;
        let (mut prefix_path, mut prefix_reason) =
            match core_llm::prefix_path_before_lookup(prefix_on, multimodal || gemma4_mm_request) {
                ("miss", _) if !prefix_route => (
                    "bypassed",
                    Some("the reference decode loop is the parity oracle and always prefills cold"),
                ),
                shared => shared,
            };
        let prefix_mtp = matches!(
            route,
            DecodeRoute::Engine {
                proposer: ProposerKind::Mtp,
                ..
            }
        );
        let prefix_boundary = match &self.model {
            Decoder::Qwen35(_) if prefix_on && prefix_route => {
                self.conversation_boundary(messages, req, &prompt_ids)
            }
            _ => None,
        };
        let snapshot_bytes = self
            .prefix_snapshot_bytes(
                prompt_ids.len().saturating_add(req.max_new_tokens as usize),
                prefix_boundary,
                prefix_mtp,
            )
            .ok_or_else(|| CoreError::InvalidRequest("prefix snapshot estimate overflow".into()))?;
        let available = request_available_memory(self.model.device())?;
        // Held prefix-cache entries are reclaimable memory (E7): a request that would not fit
        // evicts them least-recently-used first, and the snapshot it would leave behind is
        // admitted with it — or not taken, so caching never makes a request fail and never takes
        // memory admission did not grant.
        let prefix_admission = if prefix_on && prefix_route {
            self.prefix_cache()
                .admit(required, snapshot_bytes, available)
        } else {
            core_llm::PrefixAdmission {
                available: self.prefix_cache().reclaim_for(required, available),
                snapshot: false,
            }
        };
        let keep_prefix = prefix_admission.snapshot;
        if prefix_on && prefix_route && !keep_prefix {
            prefix_reason = Some(PREFIX_NOT_ADMITTED);
        }
        core_llm::admit_request_memory_with_geometry(
            admitted_prompt,
            req.max_new_tokens,
            self.descriptor.capabilities.max_context_tokens,
            if keep_prefix {
                required.saturating_add(snapshot_bytes)
            } else {
                required
            },
            prefix_admission.available,
        )?;
        let prefix_boundary = prefix_boundary.filter(|_| keep_prefix);
        let mut prefix_hit = 0usize;

        self.model
            .device()
            .synchronize()
            .map_err(|e| to_core(e.into()))?;
        let generation_started = std::time::Instant::now();
        let request_span = RequestSpan::begin();
        // The reference loop's forwards are measured, not inferred (sc-24129).
        let counted = CountingDecode::new(&self.model);
        let mut extra_forwards = 0u64;

        // Encode + splice the visuals and compute M-RoPE positions (the placeholder-expanded prompt
        // becomes the effective sequence). `None` on the text-only path.
        let mm = if multimodal && !gemma4_mm_request {
            Some(self.prepare_multimodal(&prompt_ids, &req.messages)?)
        } else {
            None
        };
        let g4 = if gemma4_mm_request {
            Some(self.prepare_gemma4(&prompt_ids, &req.messages)?)
        } else {
            None
        };
        let prompt_len = mm
            .as_ref()
            .map(|m| m.expanded_ids.len())
            .or_else(|| g4.as_ref().map(|m| m.expanded_ids.len()))
            .unwrap_or(prompt_ids.len());
        validate_context_window(
            self.descriptor.capabilities.max_context_tokens,
            prompt_len,
            req.max_new_tokens,
        )?;
        // A Gemma 4 prompt's soft-token expansion is known only now: a `draft_model` route checks
        // the draft's context window again against the effective prompt (E2). Nothing has been
        // allocated for the route yet; the priced draft is only an over-estimate.
        if let DecodeRoute::Engine {
            proposer: ProposerKind::DraftModel,
            drafts,
        } = route
        {
            let fit = core_llm::fit_draft_context(
                core_llm::SpeculativeResolution {
                    plan: SpeculativePlan::Run {
                        proposer: SpeculativeProposer::DraftModel,
                        depth: drafts,
                    },
                    fallback: None,
                },
                &self.descriptor.capabilities,
                self.draft_context(),
                prompt_len,
                req.max_new_tokens,
            );
            if let Some(why) = fit.fallback {
                let (refit, route_fallback) = self.decode_route(fit.plan);
                route = refit;
                fallbacks.push(why);
                fallbacks.extend(route_fallback);
            }
        }

        let config = GenerationConfig {
            max_new_tokens: req.max_new_tokens as usize,
            sampling: SamplingParams::from(&req.sampling),
            seed: req.seed,
            stop_tokens: self.stop_tokens.clone(),
        };
        // Structured-output constraint: build a JSON mask over the cached decode table.
        let constraint_starts_in_reasoning =
            self.descriptor.capabilities.supports_thinking && prompt_opens_thinking(&prompt);
        let mut json_mask = match req.constraint {
            Some(Constraint::Json) => {
                let table = self
                    .constraint_table
                    .get_or_init(|| self.tokenizer.constraint_decode_table());
                Some(JsonMask::new(
                    table,
                    self.stop_tokens.iter().map(|&i| i as u32),
                    constraint_starts_in_reasoning,
                ))
            }
            None => None,
        };

        let mut stop_matcher = StopMatcher::new(req.stop.iter().cloned());
        let stop_active = !stop_matcher.is_empty();
        let halt = std::cell::Cell::new(false);

        // A reasoning segmenter when the model advertises a thinking mode: it splits the decoded
        // stream into `<think>…</think>` reasoning vs answer (markers stripped) across the Thinking /
        // Content channels. `None` otherwise, so a non-thinking provider stays on the original
        // single-channel path (byte-identical streaming).
        let thinking_active = self.descriptor.capabilities.supports_thinking;
        let mut segmenter = thinking_active.then(ThinkingSegmenter::default);
        // Some chat templates open the reasoning block *in the prompt* (e.g. a Qwen3 generation
        // prompt ending `…<|im_start|>assistant\n<think>\n`), so the model generates inside the block
        // and only emits the closing `</think>`. Prime the segmenter into the Thinking channel by
        // feeding it that already-rendered opening marker (stripped, emits nothing); a Disabled
        // request renders a *closed* `<think></think>`, so this correctly does not prime.
        if let Some(seg) = segmenter.as_mut() {
            if constraint_starts_in_reasoning {
                let _ = seg.push("<think>");
            }
        }
        // A tool-call segmenter when the request offers tools and the model's template renders them:
        // it lifts `<tool_call>` blocks out of the answer channel (markup excluded from the streamed
        // text) and parses them into structured calls (story 7636). `None` otherwise, so a no-tools
        // request flows straight through `tool_pieces` unchanged.
        let tools_active = self.descriptor.capabilities.supports_tools && !req.tools.is_empty();
        let mut tool_seg = tools_active.then(|| ToolCallSegmenter::new(&req.tools));
        // Reasoning text (Thinking channel) and the answer (Content channel), accumulated as the
        // segmenter releases each span; the answer becomes the result text when thinking is active.
        let mut thinking_buf = String::new();
        let mut streamed = String::new();
        // Contract token index over *emitted* events, not the raw decode step — stripped
        // `<think>`/`</think>` marker tokens produce no event, so this stays gap-free (and equals the
        // step in the common one-delta-per-token, non-thinking case).
        let mut emit_index = 0usize;
        let mut last_id = 0u32; // id of the last emitted token, for the flushed-tail events

        // Drive the internal loop; translate token-id events to contract text-delta events via
        // incremental detokenization (re-decode the running sequence, emit the new suffix). The
        // `IncrementalDetok` guard holds back lossy U+FFFD placeholders so a multi-byte character
        // split across BPE tokens streams intact (and never panics a mid-char slice) — sc-12452.
        // The segmenter (when active) splits each delta into reasoning vs answer.
        let tokenizer = &self.tokenizer;
        let mut mtp_stats = None;
        let mut engine_record: Option<DecodeRecord> = None;
        let phase_prefill;
        let phase_decode_started;
        let out = {
            let mut acc: Vec<u32> = Vec::new();
            let mut detok = IncrementalDetok::new();
            let mut sink = |ev: StreamEvent| {
                if let StreamEvent::Token { id, step } = ev {
                    let id = id as u32;
                    acc.push(id);
                    if let Ok(text) = tokenizer.decode(&acc, true) {
                        if let Some(delta) = detok.push(&text) {
                            let delta = delta.to_string();
                            match segmenter.as_mut() {
                                Some(seg) => {
                                    for span in seg.push(&delta) {
                                        match span.channel {
                                            // Reasoning streams straight out (markers already stripped).
                                            Channel::Thinking => {
                                                thinking_buf.push_str(&span.text);
                                                last_id = id;
                                                on_event(CoreEvent::Token {
                                                    id,
                                                    text: span.text,
                                                    index: emit_index,
                                                    channel: Channel::Thinking,
                                                });
                                                emit_index += 1;
                                            }
                                            // Answer text → tool segmenter (lifts out tool-call
                                            // blocks) → emit.
                                            Channel::Content => {
                                                for piece in tool_pieces(&mut tool_seg, &span.text)
                                                {
                                                    emit_content(
                                                        &piece,
                                                        id,
                                                        (&mut stop_matcher, &halt),
                                                        &mut streamed,
                                                        &mut emit_index,
                                                        &mut last_id,
                                                        &mut *on_event,
                                                    );
                                                }
                                            }
                                        }
                                    }
                                }
                                None if tool_seg.is_some() || stop_active => {
                                    // No reasoning split, but tools are active: route the whole delta
                                    // through the tool segmenter (the answer is the only channel).
                                    for piece in tool_pieces(&mut tool_seg, &delta) {
                                        emit_content(
                                            &piece,
                                            id,
                                            (&mut stop_matcher, &halt),
                                            &mut streamed,
                                            &mut emit_index,
                                            &mut last_id,
                                            &mut *on_event,
                                        );
                                    }
                                }
                                None => {
                                    // Neither reasoning nor tools: the original single-channel path,
                                    // byte-identical to before either feature existed (raw `step`
                                    // index, no streamed accumulation).
                                    on_event(CoreEvent::Token {
                                        id,
                                        text: delta,
                                        index: step,
                                        channel: Channel::Content,
                                    });
                                }
                            }
                        }
                    }
                }
            };
            let should_stop = || halt.get();
            let should_stop_opt = stop_active.then_some(&should_stop as &dyn Fn() -> bool);
            if let (
                DecodeRoute::Engine {
                    proposer: route_proposer,
                    drafts,
                },
                Decoder::Qwen35(target),
            ) = (route, &self.model)
            {
                // The Qwen3.5 family through the unified engine over the step seam (sc-24130,
                // sc-24140, sc-24433): the plan's proposer — the MTP head or prompt lookup — or no
                // proposer, the token-at-a-time loop on the same static KV cache, checkpoint ring
                // and CUDA-graph runner, for a request whose speculation is off. The reference
                // `Decode` loop stays selectable (`set_decode_path`) as the oracle.
                let drafts = drafts as usize;
                let mut mtp_proposer = match route_proposer {
                    ProposerKind::Mtp => {
                        Some(MtpProposer::new(self.mtp.as_ref().ok_or_else(|| {
                            CoreError::Load("MTP was advertised without a loaded predictor".into())
                        })?))
                    }
                    _ => None,
                };
                // The MTP head pairs the target's hidden rows with its drafts; prompt lookup, a
                // draft model and the plain loop do not need them.
                let speculative = mtp_proposer.is_some();
                let mut ngram_proposer = NgramProposer::default();
                let mut no_proposer = NoProposer;
                // A resident draft model (sc-24436) over the effective prompt: its cache spans the
                // expanded prompt plus the budget.
                let mut draft_proposer = match route_proposer {
                    ProposerKind::DraftModel => Some(self.draft_proposer(
                        prompt_len.saturating_add(config.max_new_tokens),
                        drafts,
                    )?),
                    _ => None,
                };
                let constraint = json_mask
                    .as_mut()
                    .map(|m| m as &mut dyn RewindableConstraintMask);
                // The CUDA-graph runner (sc-24134) wraps the target when the switch is on: it
                // replays captured decode / verify steps where it can and falls back eager with
                // a named reason otherwise (`decode_record.cuda_graphs`); the engine is the
                // same either way.
                let graphs = GraphRunner::new(target);
                let stepper: &dyn StepModel<Cache = Qwen35Cache> = if cuda_graphs_enabled() {
                    &graphs
                } else {
                    target
                };
                let run = match &mm {
                    Some(m) => {
                        // Multimodal: the fused-embedding / M-RoPE prefill is the caller's, into
                        // the engine's own step cache; an MTP predictor is warmed from the same
                        // fused rows; the continuation positions are shifted by `mrope_delta`
                        // inside the cache.
                        let capacity = m.expanded_ids.len().saturating_add(config.max_new_tokens);
                        let mut cache = target
                            .new_cache_for(capacity, drafts)
                            .map_err(|e| self.request_error(e, prompt_len, req.max_new_tokens))?;
                        let (t, h, w, delta) = &m.positions;
                        let positions = [t.as_slice(), h.as_slice(), w.as_slice()];
                        let (logits, hidden) = target
                            .prefill_from_embeds_deepstack_with_hidden(
                                &m.embeds,
                                positions,
                                &mut cache,
                                &m.visual_pos_mask,
                                &m.deepstack,
                            )
                            .map_err(to_core)?;
                        cache.set_rope_delta(*delta);
                        if let Some(proposer) = mtp_proposer.as_mut() {
                            proposer
                                .warm_multimodal(&m.embeds, &hidden, positions)
                                .map_err(to_core)?;
                        }
                        target
                            .device()
                            .synchronize()
                            .map_err(|e| to_core(e.into()))?;
                        let prefill = generation_started.elapsed();
                        let decode_started = std::time::Instant::now();
                        let proposer: &mut dyn Proposer =
                            match (mtp_proposer.as_mut(), draft_proposer.as_deref_mut()) {
                                (Some(mtp), _) => mtp,
                                (None, Some(draft)) => draft,
                                (None, None) if route_proposer == ProposerKind::PromptLookup => {
                                    &mut ngram_proposer
                                }
                                (None, None) => &mut no_proposer,
                            };
                        let run = generate_speculative_monitored(
                            stepper,
                            proposer,
                            SpeculativePrompt::Prefilled {
                                cache: &mut cache,
                                logits,
                                hidden: speculative.then_some(hidden),
                                history: &m.expanded_ids,
                                position_delta: *delta,
                                // The MTP head was warmed from the fused rows above; any other
                                // proposer (a draft model) warms over the expanded prompt ids.
                                warm_proposer: !speculative,
                            },
                            &config,
                            drafts,
                            &req.cancel,
                            &mut sink,
                            constraint,
                            should_stop_opt,
                            None,
                            requested_speculative,
                        )
                        .map_err(|e| self.request_error(e, prompt_len, req.max_new_tokens))?;
                        target
                            .device()
                            .synchronize()
                            .map_err(|e| to_core(e.into()))?;
                        phase_prefill = prefill;
                        phase_decode_started = decode_started;
                        run
                    }
                    None => {
                        // Text: prefill on top of the longest prefix the cross-turn cache
                        // restores (sc-24437) — its snapshot at the conversation boundary taken
                        // on the way — then the engine warms the proposer (an MTP head resumes
                        // from the head state stored at the restored boundary) and decodes.
                        target
                            .device()
                            .synchronize()
                            .map_err(|e| to_core(e.into()))?;
                        let prefill_started = generation_started;
                        let capacity = prompt_ids.len().saturating_add(config.max_new_tokens);
                        let mut cache = target
                            .new_cache_for(capacity, drafts)
                            .map_err(|e| self.request_error(e, prompt_len, req.max_new_tokens))?;
                        let restored = self
                            .prefix_cache()
                            .restore_into(&mut cache, &prompt_ids, speculative)
                            .map_err(to_core)?;
                        let prefilled = prefill_restored(
                            stepper,
                            &mut cache,
                            restored,
                            &prompt_ids,
                            prefix_boundary,
                            speculative,
                            &req.cancel,
                        )
                        .map_err(|e| self.request_error(e, prompt_len, req.max_new_tokens))?;
                        // Measured, not looked up: the positions the prefill did not feed.
                        prefix_hit = prompt_ids.len() - prefilled.fed_tokens;
                        if prefix_hit > 0 {
                            prefix_path = "hit";
                        }
                        let snapshot = prefilled.boundary;
                        if let Some(head) = mtp_proposer.take() {
                            mtp_proposer = Some(
                                head.resume_from(prefilled.mtp)
                                    .capture_at(snapshot.as_ref().map(Boundary::len)),
                            );
                        }
                        let mut prefill = None;
                        let mut decode_started = None;
                        let mut boundary = || -> crate::error::Result<()> {
                            target.device().synchronize()?;
                            prefill = Some(prefill_started.elapsed());
                            decode_started = Some(std::time::Instant::now());
                            Ok(())
                        };
                        let proposer: &mut dyn Proposer =
                            match (mtp_proposer.as_mut(), draft_proposer.as_deref_mut()) {
                                (Some(mtp), _) => mtp,
                                (None, Some(draft)) => draft,
                                (None, None) if route_proposer == ProposerKind::PromptLookup => {
                                    &mut ngram_proposer
                                }
                                (None, None) => &mut no_proposer,
                            };
                        let run = generate_speculative_monitored(
                            stepper,
                            proposer,
                            SpeculativePrompt::Prefilled {
                                cache: &mut cache,
                                logits: prefilled.logits,
                                hidden: prefilled.hidden,
                                history: &prompt_ids,
                                position_delta: 0,
                                warm_proposer: true,
                            },
                            &config,
                            drafts,
                            &req.cancel,
                            &mut sink,
                            constraint,
                            should_stop_opt,
                            Some(&mut boundary),
                            requested_speculative,
                        )
                        .map_err(|e| self.request_error(e, prompt_len, req.max_new_tokens))?;
                        target
                            .device()
                            .synchronize()
                            .map_err(|e| to_core(e.into()))?;
                        phase_prefill = prefill.unwrap_or_else(|| generation_started.elapsed());
                        phase_decode_started =
                            decode_started.unwrap_or_else(std::time::Instant::now);
                        let captured = mtp_proposer.as_mut().and_then(MtpProposer::take_captured);
                        if let Err(e) =
                            self.prefix_cache()
                                .store(&prompt_ids, &[], &cache, snapshot, captured)
                        {
                            prefix_reason = Some(prefix_store_reason(&e));
                        }
                        run
                    }
                };
                if speculative {
                    mtp_stats = Some(MtpStats {
                        proposed_tokens: u32::try_from(run.stats.proposed).unwrap_or(u32::MAX),
                        accepted_tokens: u32::try_from(run.stats.accepted).unwrap_or(u32::MAX),
                        target_forwards: u32::try_from(run.stats.forwards).unwrap_or(u32::MAX),
                    });
                }
                engine_record = Some(run.record);
                run.output
            } else if let (
                DecodeRoute::Engine {
                    proposer: route_proposer,
                    drafts,
                },
                Decoder::Causal(model),
            ) = (route, &self.model)
            {
                // The llama family (text, the Qwen3-VL DeepStack / M-RoPE prefill and the Gemma 4
                // soft-token splice) through the unified engine over the step seam, on the static
                // KV cache (sc-24138, sc-24446): the plan's proposer — prompt lookup (sc-24433) or
                // a draft model (sc-24436) — else no proposer. The family has no MTP head;
                // admission above priced the static preallocation (the widest layer's geometry
                // plus the verify overshoot, E6) and, with the runner on, its graph workspace
                // (sc-24140). The reference `Decode` loop stays selectable (`set_decode_path`) as
                // the oracle.
                if route_proposer == ProposerKind::Mtp {
                    return Err(CoreError::Load(
                        "MTP was advertised for a non-Qwen target decoder".into(),
                    ));
                }
                let drafts = drafts as usize;
                let mut ngram_proposer = NgramProposer::default();
                let mut no_proposer = NoProposer;
                // A resident draft model (sc-24436): its cache spans the effective prompt (the
                // Gemma 4 soft-token expansion included) plus the budget.
                let mut draft_proposer = match route_proposer {
                    ProposerKind::DraftModel => Some(self.draft_proposer(
                        prompt_len.saturating_add(config.max_new_tokens),
                        drafts,
                    )?),
                    _ => None,
                };
                let proposer: &mut dyn Proposer = match draft_proposer.as_deref_mut() {
                    Some(draft) => draft,
                    None if route_proposer == ProposerKind::PromptLookup => &mut ngram_proposer,
                    None => &mut no_proposer,
                };
                let constraint = json_mask
                    .as_mut()
                    .map(|m| m as &mut dyn RewindableConstraintMask);
                // The CUDA-graph runner (sc-24134) wraps the model when the switch is on, as on
                // the MTP path. With device positions staged (sc-24441, the CUDA default) a dense
                // llama-family step is capturable and the runner replays it; a decoder on host
                // positions (`positions_host_scalar`), a MoE bank that cannot capture, or a device
                // or build the runner refuses runs every step eager, and the record names why.
                let graphs = GraphRunner::new(model);
                let stepper: &dyn StepModel<Cache = StepKvCache> = if cuda_graphs_enabled() {
                    &graphs
                } else {
                    model
                };
                let run = match (&mm, &g4) {
                    (Some(m), _) => {
                        // Qwen3-VL multimodal (sc-24446): the DeepStack / interleaved-M-RoPE
                        // prefill fills the engine's own step cache (the reference prefill's
                        // stack, `CausalLm::step_prefill_mrope_deepstack`), then the cache shifts
                        // the continuation's text positions by `mrope_delta` — on the host and the
                        // device-positions step alike — and the engine decodes with the plan's
                        // proposer over the image-expanded history, as on the hybrid. Admission
                        // above priced the expanded prompt from the visual geometry.
                        let capacity = m.expanded_ids.len().saturating_add(config.max_new_tokens);
                        let mut cache = model
                            .new_cache_for(capacity, drafts)
                            .map_err(|e| self.request_error(e, prompt_len, req.max_new_tokens))?;
                        let (t, h, w, delta) = &m.positions;
                        let logits = model
                            .step_prefill_mrope_deepstack(
                                &m.embeds,
                                [t.as_slice(), h.as_slice(), w.as_slice()],
                                &mut cache,
                                &m.visual_pos_mask,
                                &m.deepstack,
                            )
                            .map_err(to_core)?;
                        cache.set_rope_delta(*delta);
                        model
                            .device()
                            .synchronize()
                            .map_err(|e| to_core(e.into()))?;
                        let prefill = generation_started.elapsed();
                        let decode_started = std::time::Instant::now();
                        let run = generate_speculative_monitored(
                            stepper,
                            proposer,
                            SpeculativePrompt::Prefilled {
                                cache: &mut cache,
                                logits,
                                hidden: None,
                                history: &m.expanded_ids,
                                position_delta: *delta,
                                warm_proposer: true,
                            },
                            &config,
                            drafts,
                            &req.cancel,
                            &mut sink,
                            constraint,
                            should_stop_opt,
                            None,
                            requested_speculative,
                        )
                        .map_err(|e| self.request_error(e, prompt_len, req.max_new_tokens))?;
                        model
                            .device()
                            .synchronize()
                            .map_err(|e| to_core(e.into()))?;
                        phase_prefill = prefill;
                        phase_decode_started = decode_started;
                        run
                    }
                    (None, Some(m)) => {
                        // Gemma 4 multimodal: the spliced embeddings prefill the step cache on
                        // ordinary causal 1-D positions (no M-RoPE, so no position shift), then the
                        // continuation decodes through the engine. The expanded prompt (soft-token
                        // spans included) is only known here, so its static preallocation is
                        // admitted here, before anything is allocated.
                        let capacity = m.expanded_ids.len().saturating_add(config.max_new_tokens);
                        let required = self
                            .spliced_request_bytes(route, m.expanded_ids.len(), req.max_new_tokens)
                            .ok_or_else(|| {
                                CoreError::InvalidRequest("request memory estimate overflow".into())
                            })?;
                        let available = request_available_memory(model.device())?;
                        core_llm::admit_request_memory_with_geometry(
                            m.expanded_ids.len(),
                            req.max_new_tokens,
                            self.descriptor.capabilities.max_context_tokens,
                            required,
                            self.prefix_cache().reclaim_for(required, available),
                        )?;
                        let mut cache = model
                            .new_cache_for(capacity, drafts)
                            .map_err(|e| self.request_error(e, prompt_len, req.max_new_tokens))?;
                        let logits = model
                            .step_prefill_from_embeds(&m.embeds, &mut cache)
                            .map_err(to_core)?;
                        model
                            .device()
                            .synchronize()
                            .map_err(|e| to_core(e.into()))?;
                        let prefill = generation_started.elapsed();
                        let decode_started = std::time::Instant::now();
                        let run = generate_speculative_monitored(
                            stepper,
                            proposer,
                            SpeculativePrompt::Prefilled {
                                cache: &mut cache,
                                logits,
                                hidden: None,
                                history: &m.expanded_ids,
                                position_delta: 0,
                                warm_proposer: true,
                            },
                            &config,
                            drafts,
                            &req.cancel,
                            &mut sink,
                            constraint,
                            should_stop_opt,
                            None,
                            requested_speculative,
                        )
                        .map_err(|e| self.request_error(e, prompt_len, req.max_new_tokens))?;
                        model
                            .device()
                            .synchronize()
                            .map_err(|e| to_core(e.into()))?;
                        phase_prefill = prefill;
                        phase_decode_started = decode_started;
                        run
                    }
                    (None, None) => {
                        // Text: prefill on top of the longest prefix the cross-turn cache
                        // restores (sc-24437), decode, then keep a copy of the request's KV.
                        model
                            .device()
                            .synchronize()
                            .map_err(|e| to_core(e.into()))?;
                        let capacity = prompt_ids.len().saturating_add(config.max_new_tokens);
                        let mut cache = model
                            .new_cache_for(capacity, drafts)
                            .map_err(|e| self.request_error(e, prompt_len, req.max_new_tokens))?;
                        let restored = self
                            .prefix_cache()
                            .restore_into(&mut cache, &prompt_ids, false)
                            .map_err(to_core)?;
                        let prefilled = prefill_restored(
                            stepper,
                            &mut cache,
                            restored,
                            &prompt_ids,
                            None,
                            false,
                            &req.cancel,
                        )
                        .map_err(|e| self.request_error(e, prompt_len, req.max_new_tokens))?;
                        // Measured, not looked up: the positions the prefill did not feed.
                        prefix_hit = prompt_ids.len() - prefilled.fed_tokens;
                        if prefix_hit > 0 {
                            prefix_path = "hit";
                        }
                        let mut prefill = None;
                        let mut decode_started = None;
                        let mut boundary = || -> crate::error::Result<()> {
                            model.device().synchronize()?;
                            prefill = Some(generation_started.elapsed());
                            decode_started = Some(std::time::Instant::now());
                            Ok(())
                        };
                        let run = generate_speculative_monitored(
                            stepper,
                            proposer,
                            SpeculativePrompt::Prefilled {
                                cache: &mut cache,
                                logits: prefilled.logits,
                                hidden: None,
                                history: &prompt_ids,
                                position_delta: 0,
                                warm_proposer: true,
                            },
                            &config,
                            drafts,
                            &req.cancel,
                            &mut sink,
                            constraint,
                            should_stop_opt,
                            Some(&mut boundary),
                            requested_speculative,
                        )
                        .map_err(|e| self.request_error(e, prompt_len, req.max_new_tokens))?;
                        model
                            .device()
                            .synchronize()
                            .map_err(|e| to_core(e.into()))?;
                        phase_prefill = prefill.unwrap_or_else(|| generation_started.elapsed());
                        phase_decode_started =
                            decode_started.unwrap_or_else(std::time::Instant::now);
                        if keep_prefix {
                            if let Err(e) = self.prefix_cache().store(
                                &prompt_ids,
                                &run.output.tokens,
                                &cache,
                                None,
                                None,
                            ) {
                                prefix_reason = Some(prefix_store_reason(&e));
                            }
                        }
                        run
                    }
                };
                engine_record = Some(run.record);
                run.output
            } else {
                let constraint = json_mask.as_mut().map(|m| m as &mut dyn ConstraintMask);
                match &mm {
                    // Multimodal on the reference loop (the parity oracle, `set_decode_path`):
                    // prefill the spliced embeds with interleaved M-RoPE, then decode the
                    // continuation (text positions shifted by `mrope_delta`) through the shared loop.
                    Some(m) => {
                        let model = self.model.as_vlm();
                        let mut cache = model.make_cache();
                        let (t, h, w, delta) = &m.positions;
                        let first = model
                            .prefill_with_deepstack(
                                &m.embeds,
                                [t.as_slice(), h.as_slice(), w.as_slice()],
                                &mut *cache,
                                &m.visual_pos_mask,
                                &m.deepstack,
                            )
                            .map_err(to_core)?;
                        model
                            .device()
                            .synchronize()
                            .map_err(|e| to_core(e.into()))?;
                        let prefill = generation_started.elapsed();
                        let decode_started = std::time::Instant::now();
                        let shifted = Shifted {
                            model,
                            delta: *delta,
                        };
                        let shifted = CountingDecode::new(&shifted);
                        shifted.note_external_forward(); // the DeepStack prefill above
                        let result = generate_from_prefill_with_stop(
                            &shifted,
                            &mut *cache,
                            first,
                            m.expanded_ids.clone(),
                            &config,
                            &req.cancel,
                            &mut sink,
                            constraint,
                            should_stop_opt,
                        )
                        .map_err(to_core)?;
                        extra_forwards = shifted.forwards();
                        model
                            .device()
                            .synchronize()
                            .map_err(|e| to_core(e.into()))?;
                        phase_prefill = prefill;
                        phase_decode_started = decode_started;
                        result
                    }
                    // Gemma 4 multimodal: prefill the spliced embeds on ordinary causal 1-D positions
                    // (no M-RoPE, so no position shift for the continuation), then decode through the
                    // shared loop against the unwrapped decoder.
                    None => match &g4 {
                        Some(m) => {
                            let model =
                                match &self.model {
                                    Decoder::Causal(c) => c,
                                    Decoder::Qwen35(_) => return Err(CoreError::Load(
                                        "gemma 4: the multimodal path requires the generic causal \
                                 decoder"
                                            .into(),
                                    )),
                                };
                            let mut cache = model.make_cache();
                            let first = model
                                .decode_logits_from_embeds(&m.embeds, &mut *cache, 0)
                                .map_err(to_core)?;
                            counted.note_external_forward();
                            self.model
                                .device()
                                .synchronize()
                                .map_err(|e| to_core(e.into()))?;
                            let prefill = generation_started.elapsed();
                            let decode_started = std::time::Instant::now();
                            let result = generate_from_prefill_with_stop(
                                &counted,
                                &mut *cache,
                                first,
                                m.expanded_ids.clone(),
                                &config,
                                &req.cancel,
                                &mut sink,
                                constraint,
                                should_stop_opt,
                            )
                            .map_err(to_core)?;
                            self.model
                                .device()
                                .synchronize()
                                .map_err(|e| to_core(e.into()))?;
                            phase_prefill = prefill;
                            phase_decode_started = decode_started;
                            result
                        }
                        None => match &self.model {
                            Decoder::Qwen35(_) => {
                                self.model
                                    .device()
                                    .synchronize()
                                    .map_err(|e| to_core(e.into()))?;
                                let mut cache = self.model.make_cache();
                                let ids =
                                    input_ids(&prompt_ids, self.model.device()).map_err(to_core)?;
                                let first = counted.step(&ids, &mut *cache, 0).map_err(to_core)?;
                                self.model
                                    .device()
                                    .synchronize()
                                    .map_err(|e| to_core(e.into()))?;
                                let prefill = generation_started.elapsed();
                                let decode_started = std::time::Instant::now();
                                let result = generate_from_prefill_with_stop(
                                    &counted,
                                    &mut *cache,
                                    first,
                                    prompt_ids.clone(),
                                    &config,
                                    &req.cancel,
                                    &mut sink,
                                    constraint,
                                    should_stop_opt,
                                )
                                .map_err(to_core)?;
                                self.model
                                    .device()
                                    .synchronize()
                                    .map_err(|e| to_core(e.into()))?;
                                phase_prefill = prefill;
                                phase_decode_started = decode_started;
                                result
                            }
                            Decoder::Causal(_) => {
                                let mut cache = self.model.make_cache();
                                let ids =
                                    input_ids(&prompt_ids, self.model.device()).map_err(to_core)?;
                                let first = counted.step(&ids, &mut *cache, 0).map_err(to_core)?;
                                self.model
                                    .device()
                                    .synchronize()
                                    .map_err(|e| to_core(e.into()))?;
                                let prefill = generation_started.elapsed();
                                let decode_started = std::time::Instant::now();
                                let result = generate_from_prefill_with_stop(
                                    &counted,
                                    &mut *cache,
                                    first,
                                    prompt_ids.clone(),
                                    &config,
                                    &req.cancel,
                                    &mut sink,
                                    constraint,
                                    should_stop_opt,
                                )
                                .map_err(to_core)?;
                                self.model
                                    .device()
                                    .synchronize()
                                    .map_err(|e| to_core(e.into()))?;
                                phase_prefill = prefill;
                                phase_decode_started = decode_started;
                                result
                            }
                        },
                    },
                }
            }
        };

        let span_counters = request_span.counters();
        // The engine's record is measured by the engine itself (the cache it ran on, the
        // proposer, the per-verify-step syncs); the request span — which also covers a caller
        // prefill — supplies the host-side counters and every tally. Both records name the
        // proposer that ran: `none` for a request whose speculation is off, including one whose
        // option resolved to no proposer (AC3, sc-24130) — the reference loop never runs one.
        // The engine counts a caller's prefill as its one prefill forward: the prefix cache's
        // boundary snapshot is taken inside it (sc-24446).
        let mut decode_record = match engine_record {
            Some(record) => record.with_request_span(&request_span),
            None => DecodeRecord::plain(
                DecodePath::Reference,
                counted.forwards() + extra_forwards,
                out.tokens.len(),
                span_counters,
            )
            .with_attn_formulation(self.model.attn_formulation())
            .with_proposer(ProposerKind::None)
            .with_span_tallies(&request_span)
            .with_cuda_graphs(reference_path_graphs(
                request_span.cuda_graphs(),
                cuda_graphs_enabled(),
            )),
        };
        decode_record.prefix_hit_tokens = prefix_hit as u64;
        decode_record.prefix_cache = prefix_path;
        decode_record.prefix_cache_reason = prefix_reason;
        *self
            .last_decode
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(decode_record);

        // End-of-generation tails, in pipeline order. First the reasoning segmenter's held-back
        // partial marker (it turned out not to begin a marker) as current-channel text — reasoning
        // straight out, answer through the tool segmenter; then the tool segmenter's own tail (a held
        // partial `<tool_call>` / an unterminated block surfaced as content).
        if let Some(seg) = segmenter.as_mut() {
            for span in seg.flush() {
                match span.channel {
                    Channel::Thinking => {
                        thinking_buf.push_str(&span.text);
                        on_event(CoreEvent::Token {
                            id: last_id,
                            text: span.text,
                            index: emit_index,
                            channel: Channel::Thinking,
                        });
                        emit_index += 1;
                    }
                    Channel::Content => {
                        for piece in tool_pieces(&mut tool_seg, &span.text) {
                            emit_content(
                                &piece,
                                last_id,
                                (&mut stop_matcher, &halt),
                                &mut streamed,
                                &mut emit_index,
                                &mut last_id,
                                &mut *on_event,
                            );
                        }
                    }
                }
            }
        }
        if let Some(ts) = tool_seg.as_mut() {
            for piece in ts.flush() {
                emit_content(
                    &piece,
                    last_id,
                    (&mut stop_matcher, &halt),
                    &mut streamed,
                    &mut emit_index,
                    &mut last_id,
                    &mut *on_event,
                );
            }
        }

        if stop_active && !halt.get() {
            let tail = stop_matcher.flush();
            if !tail.is_empty() {
                streamed.push_str(&tail);
                on_event(CoreEvent::Token {
                    id: last_id,
                    text: tail,
                    index: emit_index,
                    channel: Channel::Content,
                });
            }
        }

        // Result text: the streamed answer when thinking or tools are active (either means the
        // streamed channel is the authoritative answer, with reasoning / tool-call markup removed);
        // otherwise the original decode-all-tokens path (byte-identical to the no-feature case).
        // `streamed` accumulates only `IncrementalDetok`-released deltas, so it carries no
        // transient U+FFFD placeholders; a character truncated by end-of-generation is dropped
        // rather than surfaced as U+FFFD (sc-12452).
        // Reasoning and tool calls, if the model produced any, are reported separately (their markup
        // excluded from `text`).
        let text = if stop_active || thinking_active || tools_active {
            streamed
        } else {
            let gen_u32: Vec<u32> = out.tokens.iter().map(|&i| i as u32).collect();
            tokenizer.decode(&gen_u32, true)?
        };
        let thinking = (!thinking_buf.is_empty()).then_some(thinking_buf);
        let tool_calls = tool_seg.map(|mut ts| ts.take_calls()).unwrap_or_default();
        let finish = if halt.get() {
            core_llm::FinishReason::Stop
        } else {
            map_finish(out.finish_reason)
        };
        let usage = Usage {
            prompt_tokens: prompt_len as u32,
            generated_tokens: out.tokens.len() as u32,
        };
        on_event(CoreEvent::Done {
            finish_reason: finish,
            usage,
        });
        Ok(TextLlmOutput {
            timings: Some(GenerationTimings {
                prefill: phase_prefill,
                decode: phase_decode_started.elapsed(),
            }),
            text,
            thinking,
            tool_calls,
            usage,
            mtp: mtp_stats,
            decode: Some(request_report(&decode_record, cuda_graphs_on, fallbacks)),
            finish_reason: Some(finish),
            kv_cache: Some(kv_cache),
        })
    }
}

/// The request's [`core_llm::DecodeReport`]: the decode record's report under the load's graph
/// switch, carrying the speculative resolution's and the route's fallback reasons verbatim (epic
/// sc-24432 E2) — the record is the engine's and starts with none.
pub(super) fn request_report(
    record: &DecodeRecord,
    cuda_graphs_on: bool,
    fallbacks: Vec<String>,
) -> core_llm::DecodeReport {
    core_llm::DecodeReport {
        fallbacks,
        ..record.report(cuda_graphs_on)
    }
}

pub(super) fn validate_context_window(
    cap: usize,
    prompt_tokens: usize,
    max_new_tokens: u32,
) -> CoreResult<()> {
    if cap == 0 {
        return Ok(());
    }
    let total = prompt_tokens
        .checked_add(max_new_tokens as usize)
        .ok_or_else(|| CoreError::InvalidRequest("prompt + generation length overflow".into()))?;
    if total > cap {
        return Err(CoreError::InvalidRequest(format!(
            "expanded prompt ({prompt_tokens} tokens) + requested generation ({max_new_tokens}) exceeds context window {cap}"
        )));
    }
    Ok(())
}

pub(super) fn map_finish(f: FinishReason) -> CoreFinish {
    match f {
        FinishReason::StopToken | FinishReason::Stopped => CoreFinish::Stop,
        FinishReason::MaxTokens => CoreFinish::Length,
        FinishReason::Cancelled => CoreFinish::Cancelled,
    }
}
