//! Prompt enhancement: the uncensored Gemma-3 enhancer, the LTX-2.5 Gemma-4 enhancer and its
//! sealed snapshot validation.

use super::*;

impl Ltx {
    /// Prompt enhancement (sc-2845). Returns the rewritten prompt when `req.enhance_prompt` is set and
    /// the enhancer produces non-empty output; `None` (use the original prompt) when off, or — matching
    /// `generate_av.py`'s try/except — on **any** enhancer failure or empty output. Failures are logged
    /// to stderr with the reference's `ENHANCER_FALLBACK:` token (success with `ENHANCED_PROMPT:`)
    /// and returned as the fallback reason, which the render reports as a warning (sc-27451).
    pub(super) fn maybe_enhance(
        &self,
        te: &LtxTextEncoder,
        req: &GenerationRequest,
    ) -> (Option<String>, Option<String>) {
        if !req.enhance_prompt {
            return (None, None);
        }
        match self.run_enhance(te, req) {
            Ok(p) if !p.trim().is_empty() => {
                eprintln!("ENHANCED_PROMPT:{p}");
                (Some(p), None)
            }
            Ok(_) => {
                eprintln!("ENHANCER_FALLBACK:EmptyOutput:prompt enhancer returned empty output");
                (
                    None,
                    Some("prompt enhancer returned empty output".to_owned()),
                )
            }
            Err(e) => {
                eprintln!("ENHANCER_FALLBACK:{e}");
                (None, Some(e.to_string()))
            }
        }
    }

    /// Run the configured enhancer: the uncensored 4-bit Gemma (`use_uncensored_enhancer`) or the
    /// already-loaded text-encoder backbone. I2V (a `Reference` image present) selects the I2V system
    /// prompt **only on the uncensored path** — the reference's censored `enhance_t2v` always uses the
    /// T2V system prompt (`generate_av.py` never calls `enhance_i2v` there), which we match.
    pub(super) fn run_enhance(
        &self,
        te: &LtxTextEncoder,
        req: &GenerationRequest,
    ) -> Result<String> {
        let is_i2v = req
            .conditioning
            .iter()
            .any(|c| matches!(c, Conditioning::Reference { .. }));
        let temperature = req
            .enhance_temperature
            .unwrap_or(enhance::DEFAULT_TEMPERATURE);
        let cfg = EnhanceConfig {
            // F-012 twin: clamp the request-supplied budget so a huge `enhance_max_tokens` can't turn
            // a single `enhance_prompt=true` request into an effectively unbounded decode job.
            max_tokens: enhance::clamp_max_tokens(req.enhance_max_tokens),
            seed: req.seed.unwrap_or(enhance::DEFAULT_SEED),
        };
        if req.use_uncensored_enhancer {
            let (model, tokenizer) = self.load_uncensored_enhancer()?;
            let system = if is_i2v {
                enhance::I2V_SYSTEM_PROMPT
            } else {
                enhance::T2V_SYSTEM_PROMPT
            };
            enhance::enhance(
                &model,
                &tokenizer,
                system,
                &req.prompt,
                &cfg,
                &SampleParams::uncensored(temperature),
                Some(&req.cancel),
            )
        } else {
            let Tokenizer::Gemma3(tokenizer) = &self.tokenizer else {
                unreachable!("Gemma-4 routes are filtered before prompt enhancement")
            };
            enhance::enhance(
                te.gemma(),
                tokenizer,
                enhance::T2V_SYSTEM_PROMPT,
                &req.prompt,
                &cfg,
                &SampleParams::censored(temperature),
                Some(&req.cancel),
            )
        }
    }

    /// Load the separate uncensored 4-bit Gemma enhancer + its tokenizer on demand (the reference
    /// `enhance_with_model` loads it per call). A standalone mlx_lm checkpoint → `model.` key prefix;
    /// its `config.json` `quantization` block drives the 4-bit dequant. The snapshot dir was staged by
    /// the caller in the `uncensored_enhancer` component at load (sc-13664); a request that asks for the
    /// uncensored enhancer without it provisioned is an actionable error here (no env / HF-cache scan).
    pub(super) fn load_uncensored_enhancer(&self) -> Result<(GemmaModel, LtxTokenizer)> {
        let dir = self.uncensored_enhancer.as_ref().ok_or_else(|| {
            Error::Msg(
                "ltx_2_3: use_uncensored_enhancer was requested but no 'uncensored_enhancer' \
                 component was staged in LoadSpec::components — provision the amoral 4-bit Gemma \
                 snapshot dir (TheCluster/amoral-gemma-3-12B-v2-mlx-4bit) via \
                 with_component(\"uncensored_enhancer\", WeightsSource::Dir(...))"
                    .into(),
            )
        })?;
        let w = Weights::from_dir(dir)?;
        let quant = resolve_gemma_quant(dir)?;
        let model =
            GemmaModel::from_weights_with_prefix(&w, GemmaConfig::gemma_3_12b(), quant, "model.")?;
        w.materialize_accessed()?;
        let tokenizer = LtxTokenizer::from_dir(dir)?;
        Ok((model, tokenizer))
    }

    /// Load and validate the stock `google/gemma-4-12B-it` snapshot staged as `enhancer/` in the
    /// LTX-2.5 rehost.  This is intentionally demand-loaded: an ordinary request never needs its
    /// extra model, while an enhanced request gets a precise local remediation instead of a hidden
    /// cache lookup or fallback to the packed LTX text encoder.
    pub(super) fn load_ltx25_enhancer(&self) -> Result<(CausalLm, TextTokenizer, Gemma4Mm, usize)> {
        let TextAssets::Gemma4 { enhancer, .. } = &self.text_assets else {
            return Err(Error::Msg(
                "ltx_2_5 enhancer selected on a non-2.5 provider route".into(),
            ));
        };
        let required = ["config.json", "tokenizer.json", "tokenizer_config.json"];
        for file in required {
            let path = enhancer.join(file);
            if !path.is_file() {
                return Err(Error::Msg(format!(
                    "ltx_2_5: prompt enhancement requires the offline google/gemma-4-12B-it \
                     snapshot staged at {}; missing {file}. No job-time download is attempted",
                    enhancer.display()
                )));
            }
        }
        let (cfg, config_json, processor_json) = validate_ltx25_enhancer_snapshot(enhancer)?;
        let vocab_size = cfg.vocab_size as usize;
        let weights = LlmWeights::from_dir(enhancer).map_err(|e| {
            Error::Msg(format!(
                "ltx_2_5: staged enhancer weights at {} are unusable: {e}",
                enhancer.display()
            ))
        })?;
        let layout = Gemma4Layout::detect(&weights).map_err(|e| {
            Error::Msg(format!(
                "ltx_2_5: staged enhancer at {} has no recognized Gemma-4 tensor layout: {e}",
                enhancer.display()
            ))
        })?;
        let mm_cfg = Gemma4MmConfig::from_json(&config_json, processor_json.as_ref())
            .map_err(|e| Error::Msg(format!("ltx_2_5: invalid Gemma-4 multimodal config: {e}")))?;
        let mm = Gemma4Mm::from_weights(&weights, layout, mm_cfg)
            .map_err(|e| Error::Msg(format!("ltx_2_5: invalid Gemma-4 vision weights: {e}")))?;
        if mm.vision.is_none() {
            return Err(Error::Msg(format!(
                "ltx_2_5: enhancer at {} has no complete Gemma-4 vision/projector tensors; I2V \
                 prompt enhancement cannot be advertised",
                enhancer.display()
            )));
        }
        let model = CausalLm::from_weights(&weights, "", cfg).map_err(|e| {
            Error::Msg(format!(
                "ltx_2_5: staged enhancer at {} does not match its Gemma-4 config: {e}",
                enhancer.display()
            ))
        })?;
        let tokenizer = TextTokenizer::from_file(
            enhancer.join("tokenizer.json"),
            TokenizerConfig {
                max_length: usize::MAX,
                pad_token_id: 0,
                chat_template: ChatTemplate::None,
                pad_to_max_length: false,
            },
        )?;
        Ok((model, tokenizer, mm, vocab_size))
    }

    /// Run stock-Gemma enhancement with the snapshot's own template, then make that exact returned
    /// string the only prompt fed to LTX's conditioning encoder below in `stage_text_phase`.
    pub(super) fn maybe_enhance_gemma4(&self, req: &GenerationRequest) -> Result<Option<String>> {
        if !req.enhance_prompt {
            return Ok(None);
        }
        if req.use_uncensored_enhancer {
            return Err(Error::Unsupported(
                "ltx_2_5: use_uncensored_enhancer is unsupported; this route uses only the \
                 licensed stock google/gemma-4-12B-it enhancer staged at enhancer/"
                    .into(),
            ));
        }
        let (model, tokenizer, mm, vocab_size) = self.load_ltx25_enhancer()?;
        let reference = req
            .conditioning
            .iter()
            .find_map(|condition| match condition {
                Conditioning::Reference { image, .. } => Some(image),
                _ => None,
            });
        let TextAssets::Gemma4 {
            enhancer_template, ..
        } = &self.text_assets
        else {
            unreachable!("Gemma-4 enhancement only runs on the 2.5 text-assets route")
        };
        let (system, user) = ltx25_enhancer_turn(&req.prompt, reference.is_some());
        let rendered = enhancer_template
            .render(&[Message::system(system), Message::user(user)], true)
            .map_err(|e| Error::Msg(format!("ltx_2_5: enhancer chat template mismatch: {e}")))?;
        let prompt_ids = tokenizer.encode_ids(&rendered, false)?;
        let prefill = match reference {
            Some(image) => ltx25_i2v_enhancer_prefill(&model, &mm, prompt_ids, image)?,
            None => enhance::Gemma4EnhancePrefill::Text(prompt_ids),
        };
        let cfg = EnhanceConfig {
            max_tokens: enhance::clamp_gemma4_max_tokens(req.enhance_max_tokens),
            seed: req.seed.unwrap_or(enhance::GEMMA4_DEFAULT_SEED),
        };
        let cache = self.enhancer_cache.as_ref().ok_or_else(|| {
            Error::Msg("ltx_2_5: Gemma-4 enhancer cache is absent from the 2.5 route".into())
        })?;
        let (prompt, _decode) = enhance::enhance_gemma4(
            &model,
            &tokenizer,
            prefill,
            &cfg,
            vocab_size,
            &req.cancel,
            &mut cache.borrow_mut(),
        )?;
        if prompt.trim().is_empty() {
            return Err(Error::Msg(
                "ltx_2_5: Gemma-4 enhancer returned an empty prompt; refusing to silently use the \
                 original prompt"
                    .into(),
            ));
        }
        eprintln!("ENHANCED_PROMPT:{prompt}");
        Ok(Some(prompt))
    }
}

/// Exact v1.2.0 Gemma-4 conversation payload. The image marker occupies the image content item in
/// the upstream multimodal message; the stock tokenizer maps it to `image_token_id`, which the
/// prefill path expands and replaces with projected patch rows.
pub(super) fn ltx25_enhancer_turn(prompt: &str, is_i2v: bool) -> (&'static str, String) {
    if is_i2v {
        (
            enhance::GEMMA4_I2V_SYSTEM_PROMPT,
            format!("<|image|>User Raw Input Prompt: {prompt}."),
        )
    } else {
        (
            enhance::GEMMA4_T2V_SYSTEM_PROMPT,
            format!("user prompt: {prompt}"),
        )
    }
}

/// Prepare the actual Gemma-4 I2V prefill: resize to an aspect-preserving soft-token grid,
/// channel-major patchify, run `patch_dense` + norms/position table + multimodal projector, expand
/// the single image marker to `boi + image_token*N + eoi`, and splice those projected rows into the
/// decoder input embeddings.
pub(super) fn ltx25_i2v_enhancer_prefill(
    model: &CausalLm,
    mm: &Gemma4Mm,
    prompt_ids: Vec<i32>,
    image: &Image,
) -> Result<enhance::Gemma4EnhancePrefill> {
    let tower = mm.vision.as_ref().ok_or_else(|| {
        Error::Msg("ltx_2_5: Gemma-4 enhancer has no vision/projector path for I2V".into())
    })?;
    let expected = image.width as usize * image.height as usize * 3;
    if image.pixels.len() != expected {
        return Err(Error::Msg(format!(
            "ltx_2_5: I2V enhancer image {}x{} needs {expected} RGB bytes, got {}",
            image.width,
            image.height,
            image.pixels.len()
        )));
    }
    let cfg = tower.config();
    let (gh, gw) = gemma4_mm::soft_token_grid(
        image.width as usize,
        image.height as usize,
        cfg.max_soft_tokens,
    );
    let (target_w, target_h) = (gw * cfg.patch_pixels, gh * cfg.patch_pixels);
    let resized = mlx_llm::image::resize_bicubic_u8(
        &image.pixels,
        image.height as usize,
        image.width as usize,
        target_h,
        target_w,
    )
    .map_err(|e| Error::Msg(format!("ltx_2_5: I2V enhancer image resize: {e}")))?;
    let flat = gemma4_mm::patchify(&resized, target_w, target_h, cfg.patch_pixels)
        .map_err(|e| Error::Msg(format!("ltx_2_5: I2V enhancer patchify: {e}")))?;
    let patches = gemma4_mm::patch_array(&flat, cfg.patch_elems())
        .map_err(|e| Error::Msg(format!("ltx_2_5: I2V enhancer patch tensor: {e}")))?;
    let features = tower
        .forward(&patches, (gh, gw))
        .map_err(|e| Error::Msg(format!("ltx_2_5: I2V enhancer vision/projector: {e}")))?;
    let expanded = gemma4_mm::expand_framed_placeholders(
        &prompt_ids,
        mm.cfg.image_token_id,
        mm.cfg.boi_token_id,
        mm.cfg.eoi_token_id,
        &[gh * gw],
    )
    .map_err(|e| Error::Msg(format!("ltx_2_5: I2V enhancer image marker: {e}")))?;
    let ids = Array::from_slice(&expanded, &[1, expanded.len() as i32]);
    let embeds = model
        .embed_input_ids(&ids)
        .and_then(|embeds| {
            model.splice_image_features(&embeds, &expanded, &features, mm.cfg.image_token_id)
        })
        .map_err(|e| Error::Msg(format!("ltx_2_5: I2V enhancer embedding splice: {e}")))?;
    Ok(enhance::Gemma4EnhancePrefill::Multimodal {
        input_ids: expanded,
        embeds,
    })
}

/// Build the enhancer chat renderer from the LTX-2.5 text encoder's packed
/// `hf_asset__chat_template.jinja`. This is the story's single template authority; the separately
/// staged stock decoder contributes weights/tokenizer/processor, not a competing conversation
/// format.
pub(super) fn ltx25_packed_chat_template(assets: &GemmaAssets) -> Result<JinjaChatTemplate> {
    let tokenizer_config: serde_json::Value =
        serde_json::from_slice(assets.sidecar("tokenizer_config.json")?).map_err(|e| {
            Error::Msg(format!(
                "ltx_2_5: packed enhancer tokenizer_config.json is invalid: {e}"
            ))
        })?;
    let token = |name: &str| match tokenizer_config.get(name) {
        Some(serde_json::Value::String(value)) => value.clone(),
        Some(serde_json::Value::Object(value)) => value
            .get("content")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default()
            .to_string(),
        _ => String::new(),
    };
    let source = assets.sidecar_str("chat_template.jinja")?;
    if source.trim().is_empty() {
        return Err(Error::Msg(
            "ltx_2_5: packed hf_asset__chat_template.jinja is empty".into(),
        ));
    }
    Ok(JinjaChatTemplate::with_tokens(
        source,
        token("bos_token"),
        token("eos_token"),
    ))
}

pub(super) fn manifest_string<'a>(manifest: &'a serde_json::Value, key: &str) -> Result<&'a str> {
    manifest
        .get(key)
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| Error::Msg(format!("ltx_2_5: enhancer manifest needs string `{key}`")))
}

/// Recompute the manifest's semantic digest independent of JSON whitespace/key order.
pub(super) fn ltx25_enhancer_inventory_sha256(manifest: &serde_json::Value) -> Result<String> {
    let schema = manifest_string(manifest, "schema_version")?;
    let repository = manifest_string(manifest, "source_repository")?;
    let revision = manifest_string(manifest, "source_revision")?;
    let files = manifest
        .get("files")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| Error::Msg("ltx_2_5: enhancer manifest needs object `files`".into()))?;
    let mut digest = Sha256::new();
    for value in [schema, repository, revision] {
        digest.update(value.as_bytes());
        digest.update([0]);
    }
    let mut names: Vec<&str> = files.keys().map(String::as_str).collect();
    names.sort_unstable();
    for name in names {
        let entry = &files[name];
        let bytes = entry
            .get("bytes")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| {
                Error::Msg(format!(
                    "ltx_2_5: enhancer manifest file `{name}` needs integer `bytes`"
                ))
            })?;
        let sha256 = entry
            .get("sha256")
            .and_then(serde_json::Value::as_str)
            .filter(|value| {
                value.len() == 64
                    && value
                        .bytes()
                        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            })
            .ok_or_else(|| {
                Error::Msg(format!(
                    "ltx_2_5: enhancer manifest file `{name}` needs lowercase SHA-256"
                ))
            })?;
        for value in [name.to_owned(), bytes.to_string(), sha256.to_owned()] {
            digest.update(value.as_bytes());
            digest.update([0]);
        }
    }
    Ok(format!("{:x}", digest.finalize()))
}

/// Require SC-18780's exact publication inventory before consulting any model config or mapping
/// any weights. The manifest is copied into `enhancer/` alongside the stock snapshot and is fully
/// reproducible offline from that snapshot's file sizes and SHA-256 values.
pub(super) fn validate_ltx25_enhancer_manifest(enhancer: &Path) -> Result<serde_json::Value> {
    let canonical: serde_json::Value =
        serde_json::from_str(LTX25_ENHANCER_MANIFEST).map_err(|error| {
            Error::Msg(format!(
                "ltx_2_5: embedded enhancer manifest invalid: {error}"
            ))
        })?;
    let computed = ltx25_enhancer_inventory_sha256(&canonical)?;
    if computed != LTX25_ENHANCER_INVENTORY_SHA256
        || manifest_string(&canonical, "inventory_sha256")? != computed
    {
        return Err(Error::Msg(
            "ltx_2_5: embedded enhancer inventory digest is inconsistent".into(),
        ));
    }
    if manifest_string(&canonical, "schema_version")? != LTX25_ENHANCER_SCHEMA
        || manifest_string(&canonical, "source_repository")? != LTX25_ENHANCER_REPOSITORY
        || manifest_string(&canonical, "source_revision")? != LTX25_ENHANCER_REVISION
    {
        return Err(Error::Msg(
            "ltx_2_5: embedded enhancer source identity is inconsistent".into(),
        ));
    }

    let path = enhancer.join(LTX25_ENHANCER_MANIFEST_FILE);
    let bytes = std::fs::read(&path).map_err(|error| {
        Error::Msg(format!(
            "ltx_2_5: prompt enhancement requires the offline {LTX25_ENHANCER_REPOSITORY} \
             snapshot plus {LTX25_ENHANCER_MANIFEST_FILE} staged at {}; {error}. No job-time \
             download is attempted",
            enhancer.display()
        ))
    })?;
    let staged: serde_json::Value = serde_json::from_slice(&bytes).map_err(|error| {
        Error::Msg(format!(
            "ltx_2_5: enhancer asset manifest {} is invalid JSON: {error}",
            path.display()
        ))
    })?;
    if staged != canonical
        || manifest_string(&staged, "inventory_sha256")? != LTX25_ENHANCER_INVENTORY_SHA256
        || ltx25_enhancer_inventory_sha256(&staged)? != LTX25_ENHANCER_INVENTORY_SHA256
    {
        return Err(Error::Msg(format!(
            "ltx_2_5: enhancer asset manifest {} does not identify exact \
             {LTX25_ENHANCER_REPOSITORY}@{LTX25_ENHANCER_REVISION}",
            path.display()
        )));
    }
    Ok(staged)
}

pub(super) fn validate_ltx25_enhancer_file(
    enhancer: &Path,
    manifest: &serde_json::Value,
    name: &str,
) -> Result<()> {
    let entry = manifest
        .get("files")
        .and_then(serde_json::Value::as_object)
        .and_then(|files| files.get(name))
        .ok_or_else(|| Error::Msg(format!("ltx_2_5: enhancer manifest is missing `{name}`")))?;
    let expected_bytes = entry
        .get("bytes")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| Error::Msg(format!("ltx_2_5: enhancer manifest `{name}` has no size")))?;
    let expected_sha256 = entry
        .get("sha256")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| Error::Msg(format!("ltx_2_5: enhancer manifest `{name}` has no digest")))?;
    let path = enhancer.join(name);
    if !path.is_file() {
        return Err(Error::Msg(format!(
            "ltx_2_5: exact stock enhancer is missing `{name}` at {}",
            enhancer.display()
        )));
    }
    // Size first, so a wrong-size file is refused without reading its ~22 GiB.
    let size = std::fs::metadata(&path)?.len();
    if size != expected_bytes {
        return Err(Error::Msg(format!(
            "ltx_2_5: enhancer asset `{name}` is {size} bytes, expected {expected_bytes} for \
             {LTX25_ENHANCER_REPOSITORY}@{LTX25_ENHANCER_REVISION}"
        )));
    }
    // SHA-256 once per immutable file identity (shared, single-flight `ArtifactSeal` memo), so
    // repeated enhanced requests do not re-read the stock model.
    let actual = ArtifactSeal::pin(&path)?.content_sha256_hex();
    if actual != expected_sha256 {
        return Err(Error::Msg(format!(
            "ltx_2_5: enhancer asset `{name}` digest mismatch (got {actual}, expected \
             {expected_sha256} for {LTX25_ENHANCER_REPOSITORY}@{LTX25_ENHANCER_REVISION})"
        )));
    }
    Ok(())
}

/// Validate the immutable, offline asset identity before `Weights::from_dir` maps any tensor.
/// A version/config/template mismatch is a request error rather than a fallback to a different
/// model or chat format.  Kept free-standing so the weights-free provider tests exercise the same
/// gate the ordinary route executes.
pub(super) fn validate_ltx25_enhancer_snapshot(
    enhancer: &Path,
) -> Result<(ModelConfig, serde_json::Value, Option<serde_json::Value>)> {
    let manifest = validate_ltx25_enhancer_manifest(enhancer)?;
    // Config first: a matching-but-wrong Gemma-4 snapshot fails on its exact config digest without
    // ever touching the 22 GiB model file. The remaining complete inventory is still verified below
    // before `load_ltx25_enhancer` reaches `LlmWeights::from_dir`.
    validate_ltx25_enhancer_file(enhancer, &manifest, "config.json")?;
    let config_text = std::fs::read_to_string(enhancer.join("config.json"))?;
    let config_json: serde_json::Value = serde_json::from_str(&config_text).map_err(|e| {
        Error::Msg(format!(
            "ltx_2_5: enhancer config {} is not valid JSON: {e}",
            enhancer.join("config.json").display()
        ))
    })?;
    let model_type = config_json
        .get("model_type")
        .and_then(serde_json::Value::as_str);
    if model_type != Some("gemma4_unified")
        || config_json.get("architectures")
            != Some(&serde_json::json!([
                "Gemma4UnifiedForConditionalGeneration"
            ]))
    {
        return Err(Error::Msg(format!(
            "ltx_2_5: enhancer at {} does not declare the exact stock Gemma-4 unified config \
             (model_type={model_type:?})",
            enhancer.display()
        )));
    }
    let cfg = ModelConfig::from_json(&config_json)
        .map_err(|e| Error::Msg(format!("ltx_2_5: invalid Gemma-4 enhancer config: {e}")))?;
    if !cfg.is_gemma4() {
        return Err(Error::Msg(format!(
            "ltx_2_5: enhancer at {} is not google/gemma-4-12B-it (expected a Gemma-4 \
             config); refusing to misreport prompt enhancement",
            enhancer.display()
        )));
    }
    if cfg.final_logit_softcap != Some(30.0) {
        return Err(Error::Msg(format!(
            "ltx_2_5: enhancer at {} declares final_logit_softcapping={:?}, expected exactly 30.0",
            enhancer.display(),
            cfg.final_logit_softcap
        )));
    }
    for file in [
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "processor_config.json",
        "chat_template.jinja",
        "model.safetensors",
    ] {
        validate_ltx25_enhancer_file(enhancer, &manifest, file)?;
    }
    let _: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
        enhancer.join("tokenizer_config.json"),
    )?)
    .map_err(|e| {
        Error::Msg(format!(
            "ltx_2_5: enhancer tokenizer_config.json is invalid: {e}"
        ))
    })?;
    let processor_json = serde_json::from_str(&std::fs::read_to_string(
        enhancer.join("processor_config.json"),
    )?)
    .map_err(|e| {
        Error::Msg(format!(
            "ltx_2_5: enhancer processor_config.json is invalid: {e}"
        ))
    })?;
    Ok((cfg, config_json, Some(processor_json)))
}
