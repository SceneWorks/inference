//! Provider load: the Gemma text-encoder / enhancer resolution and the LTX-2.3 / LTX-2.5
//! construction paths.

use super::*;

/// Locate the Gemma-3-12B text-encoder snapshot from the **required** `LoadSpec::text_encoder` slot
/// (sc-8827; required as of sc-13664). The caller provisions the TE path through the spec — there is no
/// env side-channel and no on-disk HF-cache scan any more (epic 13657): an absent slot is a load-time,
/// actionable error, and a set-but-nonexistent path errors up front. `pub(crate)` so the trainer
/// (sc-3047) resolves the TE snapshot exactly as inference does (it too must be handed a
/// `LoadSpec::text_encoder`).
pub(crate) fn resolve_gemma_dir(
    override_src: Option<&WeightsSource>,
) -> Result<std::path::PathBuf> {
    let Some(WeightsSource::Dir(p) | WeightsSource::File(p)) = override_src else {
        return Err(Error::Msg(
            "ltx_2_3 requires the Gemma-3-12B text encoder — set LoadSpec::text_encoder to the \
             gemma-3-12b-it snapshot dir (e.g. LoadSpec { text_encoder: \
             Some(WeightsSource::Dir(...)), .. }). It is no longer auto-discovered from an \
             environment variable or the HF cache."
                .into(),
        ));
    };
    if !p.exists() {
        return Err(Error::Msg(format!(
            "ltx_2_3: LoadSpec text_encoder path does not exist: {}",
            p.display()
        )));
    }
    Ok(p.clone())
}

/// Resolve the optional **uncensored** 4-bit Gemma enhancer snapshot dir (sc-2845
/// `--use-uncensored-enhancer`, reference `TheCluster/amoral-gemma-3-12B-v2-mlx-4bit`) from the
/// caller-staged `uncensored_enhancer` [`LoadSpec::components`] entry (sc-13664 — no
/// `$LTX_UNCENSORED_GEMMA_DIR`, no HF-cache scan). `None` when the caller did not provision it (the
/// feature is opt-in per request); a set-but-nonexistent path errors up front at load. A standalone
/// mlx_lm checkpoint (`model.` key prefix).
pub(super) fn resolve_uncensored_enhancer(
    component: Option<&WeightsSource>,
) -> Result<Option<std::path::PathBuf>> {
    let Some(src) = component else {
        return Ok(None);
    };
    let (WeightsSource::Dir(p) | WeightsSource::File(p)) = src;
    if !p.exists() {
        return Err(Error::Msg(format!(
            "ltx_2_3: the 'uncensored_enhancer' component path does not exist: {}",
            p.display()
        )));
    }
    Ok(Some(p.clone()))
}

/// Read the Gemma snapshot's `config.json` top-level `quantization` block — the reference TE-quant
/// trigger (`utils.apply_quantization`). `None` for the default `…-bf16` snapshot (no block). Only the
/// `affine` mode is consumed (the one `quantized_matmul`/`dequantize` implement); a non-affine mode is
/// a hard error rather than a silent mis-decode.
pub(crate) fn resolve_gemma_quant(gemma_dir: &std::path::Path) -> Result<Option<GemmaQuant>> {
    let path = gemma_dir.join("config.json");
    if !path.exists() {
        return Ok(None);
    }
    let text = mlx_gen::gen_core::__model_input_reader!(
std::fs::read_to_string(&path),
"crates/media/mlx-gen/mlx-gen-ltx/src/model/load.rs::resolve_gemma_quant::std::fs::read_to_string@3041",
ArtifactInput,
&path,
|__model_input_path| std::fs::read_to_string(__model_input_path)
)?;
    let v: serde_json::Value = serde_json::from_str(&text)
        .map_err(|e| Error::Msg(format!("ltx_2_3: parse gemma config.json: {e}")))?;
    let Some(q) = v.get("quantization") else {
        return Ok(None);
    };
    if let Some(mode) = q.get("mode").and_then(|m| m.as_str()) {
        if mode != "affine" {
            return Err(Error::Msg(format!(
                "ltx_2_3: gemma quantization mode {mode:?} is not supported (only affine)"
            )));
        }
    }
    match (
        q.get("group_size").and_then(|x| x.as_i64()),
        q.get("bits").and_then(|x| x.as_i64()),
    ) {
        (Some(g), Some(b)) => Ok(Some(GemmaQuant {
            group: g as i32,
            bits: b as i32,
        })),
        _ => Ok(None),
    }
}

/// Load the model from a split-weight snapshot directory (the `ltx_2_3_base*` tree). Reads
/// `embedded_config.json`, locates the Gemma TE separately, and assembles every component.
pub fn load(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    let root =
        match &spec.weights {
            WeightsSource::Dir(p) => p,
            WeightsSource::File(_) => return Err(Error::Msg(
                "ltx_2_3: expected a model directory (split-weight snapshot), not a single file"
                    .into(),
            )),
        };
    // Named-component contract (sc-13658/sc-13664): LTX-2.3 reads only the optional `uncensored_enhancer`
    // component; its main Gemma TE rides the typed `text_encoder` slot, not this map. Reject any
    // unrecognized component key up front, then stage the (existence-checked) enhancer path if provided.
    reject_unknown_components(spec, &["uncensored_enhancer"], MODEL_ID)?;
    let uncensored_enhancer =
        resolve_uncensored_enhancer(spec.components.get("uncensored_enhancer"))?;
    // Layout gate (sc-18757): which LTX generation this tree holds is decided by its declared
    // `model_version` — NOT by which files are present and NOT by their names. This engine
    // implements the LTX-2.3 all-in-one component set; an LTX-2.5 split bundle carries a different
    // DiT, a Gemma 4 text encoder and a diffusion VAE, so it is refused **here**, by version, with
    // the version named. Without this gate a 2.5 tree fell through to the per-file existence check
    // below and died as "missing transformer.safetensors" — a filename-shaped error that points at
    // the wrong problem. `crate::bundle` resolves the split layout for the engine that implements it.
    let declared_version = crate::bundle::declared_model_version(root)?;
    if layout_for_declared_version(declared_version.as_deref()) == LtxCheckpointLayout::Split {
        return Err(Error::Msg(format!(
            "ltx_2_3: {} declares model_version {:?}, which ships as a split-component bundle \
             (per-component transformer / text encoder / video VAE / audio VAE / duration head / \
             latent upsamplers, each with its own config). The ltx_2_3 engine loads the all-in-one \
             LTX-2.3 layout only.",
            root.display(),
            declared_version.as_deref().unwrap_or_default(),
        )));
    }
    // Quantization geometry rides on the checkpoint's `split_model.json` (sc-2686): the transformer is
    // shipped selectively quantized (Q4 for `base_q4`, Q8 for `base_q8`), bits/group from the
    // manifest — never hardcoded. The per-Linear `.scales` predicate (in `transformer.rs`) then picks
    // which Linears are quantized, matching `generate_av.py`'s `_should_quantize`.
    let split = SplitModel::from_model_dir(root)?;
    // `spec.quantize`, when set, only *asserts* the expected level. LTX can't re-quantize a dense
    // checkpoint (there is no dense LTX transformer — it ships pre-packed from the reference
    // `convert.py`, which casts f32→bf16 before quantizing), so a mismatch is a hard load error
    // rather than a silent re-quant.
    if let Some(q) = spec.quantize {
        if !split.quantized {
            return Err(Error::Msg(format!(
                "ltx_2_3: spec.quantize={q:?} but {} carries no split_model.json quant manifest — \
                 LTX quant is checkpoint-driven; point at a quantized checkpoint (e.g. ltx_2_3_base_q4)",
                root.display()
            )));
        }
        if q.bits() != split.bits {
            return Err(Error::Msg(format!(
                "ltx_2_3: spec.quantize={q:?} (bits {}) disagrees with the checkpoint's \
                 split_model.json (bits {})",
                q.bits(),
                split.bits
            )));
        }
    }
    // Precision selection. `Bf16` (the [`LoadSpec`] default) → the reference's **native** bf16
    // activations × quantized weights — the production-speed path; `Fp32` → f32 activations ×
    // quantized weights — the quality target. Both are bit-exact to their reference golden (sc-2842;
    // the distilled stage-1 sampler is chaos-sensitive, so each per-forward is bit-exact). The latent
    // statistics (the upsampler's un-/re-normalize) follow the path dtype so the whole denoise stays
    // in that precision; the VAE decode stays f32 in both — a post-sampling quality island.
    let (dit_prec, stat_dt) = match spec.precision {
        LoadPrecision::Bf16 => (
            Precision::quant_bf16(split.bits, split.group),
            Dtype::Bfloat16,
        ),
        LoadPrecision::Fp32 => (
            Precision::quant_f32(split.bits, split.group),
            Dtype::Float32,
        ),
    };

    let config = LtxConfig::from_model_dir(root)?;
    let vae_config = LtxVaeConfig::from_model_dir(root)?;
    let audio_vae_config = AudioVaeConfig::from_model_dir(root)?;
    let vocoder_config = VocoderConfig::from_model_dir(root)?;

    // Resolve the Gemma TE location + its quant (cheap — a path lookup + one `config.json` read). The
    // ~24 GB Gemma backbone and the `connector`/`transformer` weights are deliberately NOT loaded here:
    // sc-10976 defers them to `build_text_encoder`/`build_transformer`, built per-generate so the TE is
    // dropped (+ `clear_cache()`) before the DiT materializes (mirroring Wan). Fail-fast on a missing
    // component is preserved by a cheap existence check, so a broken tree still errors at load rather
    // than at first generate.
    let gemma_dir = resolve_gemma_dir(spec.text_encoder.as_ref())?;
    // Selectively quantize the Gemma backbone iff the snapshot's `config.json` says so (the reference
    // `apply_quantization` path; sc-2686). The default `…-bf16` snapshot ⇒ `None` ⇒ dense bf16 TE.
    let gemma_quant = resolve_gemma_quant(&gemma_dir)?;
    for f in ["connector.safetensors", "transformer.safetensors"] {
        if !root.join(f).exists() {
            return Err(Error::Msg(format!(
                "ltx_2_3: missing {f} in the model dir {}",
                root.display()
            )));
        }
    }
    // SC-19109: bind the canonical dense-bf16 Gemma route to the same resolved split manifest,
    // component paths and additive overlays the engine below will materialize. Quantized Gemma is a
    // supported generator route but has no matching evidence identity yet, so it deliberately loads
    // without a memory contract rather than borrowing the canonical route's measurements.
    let loaded_memory =
        crate::memory_strategy::contract_for_loaded(spec, &split, &gemma_dir, gemma_quant)?;
    let (memory_strategy, memory_tier, memory_overlay) = match loaded_memory {
        Some((contract, tier, overlay)) => (Some(contract), Some(tier), overlay),
        None => (None, None, None),
    };

    // The small components (each ≪ the TE / DiT) stay resident. The VAE *decoder* + audio VAE + vocoder
    // load here; the VAE *encoder* is still lazy on first I2V encode (F-048).
    let vae_w = Weights::from_file(root.join("vae_decoder.safetensors"))?;
    let audio_vae_w = Weights::from_file(root.join("audio_vae.safetensors"))?;
    let vocoder_w = Weights::from_file(root.join("vocoder.safetensors"))?;
    // Resident components cross the GPU-view guard here (sc-22414, `mlx_gen::coherence`).
    vae_w.materialize()?;
    audio_vae_w.materialize()?;
    vocoder_w.materialize()?;

    // Loaded through the path constructor, so a stamped checkpoint's declared config is
    // cross-checked against the structure the weights imply instead of the rank silently winning.
    let upsampler = LatentUpsampler::from_checkpoint(root.join("upsampler.safetensors"))?;
    // The VAE encoder serves I2V conditioning (sc-2685) but pure-T2V+A requests never touch it, so it
    // is loaded **lazily** on first encode — `vae_encoder.safetensors` (hundreds of MB) stays off the
    // resident set for T2V (F-048). `generate_av.py` supports I2V+Audio (image-conditioned video).
    let vae = LtxVideoVae::from_weights_lazy_encoder(
        &vae_w,
        root.join("vae_encoder.safetensors"),
        &vae_config,
    )?;
    // The audio VAE decoder + vocoder run f32 (post-sampling quality islands, gated bit-exact).
    let audio_decoder = AudioDecoder::from_weights(&audio_vae_w, &audio_vae_config)?;
    let vocoder = LtxVocoder::from_weights(&vocoder_w, &vocoder_config)?;
    let audio_sample_rate = vocoder_config.final_sample_rate() as u32;
    // The VAE `per_channel_statistics` double as the upsampler's latent norm, at the path dtype.
    let latent_mean = to_dtype(vae_w.require("per_channel_statistics.mean")?, stat_dt)?;
    let latent_std = to_dtype(vae_w.require("per_channel_statistics.std")?, stat_dt)?;
    let tokenizer = LtxTokenizer::from_dir(&gemma_dir)?;

    Ok(Box::new(Ltx {
        descriptor: descriptor(),
        memory_strategy,
        memory_tier,
        memory_overlay,
        tokenizer: Tokenizer::Gemma3(tokenizer),
        transformer_path: root.join("transformer.safetensors"),
        config,
        dit_prec,
        adapters: spec.adapters.clone(),
        text_assets: TextAssets::Gemma3 {
            dir: gemma_dir,
            quant: gemma_quant,
        },
        execution: LtxExecution::Ltx23,
        uncensored_enhancer,
        enhancer_cache: None,
        upsampler,
        vae,
        audio_decoder,
        vocoder,
        latent_mean,
        latent_std,
        audio_sample_rate,
        stat_dt,
    }))
}

/// Resolve the LTX-2.5 split layout through the ordinary provider registration. The actual tensor
/// assembly is request-scoped and invoked by [`Generator::generate`], not hidden behind a filename
/// convention or a separate loader entry point.
pub fn load_25(spec: &LoadSpec) -> Result<Box<dyn Generator>> {
    let mut known = crate::bundle::split_component_ids();
    known.push(LTX25_ENHANCER_COMPONENT);
    reject_unknown_components(spec, &known, MODEL_25_ID)?;
    let bundle = crate::bundle::resolve_split_bundle(spec)?;
    let variant = crate::dev_sampler::from_bundle(&bundle)?;
    if bundle.layout() != LtxCheckpointLayout::Split {
        return Err(Error::Msg(format!(
            "ltx_2_5: expected an LTX-2.5 split-component bundle, got {}",
            bundle.layout().id()
        )));
    }
    // These are intentional load-path checks, not descriptor-only declarations.  In particular,
    // `from_bundle` selects the transformer's own 2.5 config and the Gemma check ties that DiT to
    // the packed encoder before a generation can allocate noise.  The duration and temporal
    // components are required here because the advertised request controls execute through them.
    let config = LtxConfig::from_bundle(&bundle)?;
    if !config.use_keyframes_abs_pos_embedding {
        return Err(Error::Msg(
            "ltx_2_5: transformer lacks use_keyframes_abs_pos_embedding, so generated-keyframe \
             slots must not be advertised or executed"
                .into(),
        ));
    }
    crate::bundle::assert_gemma_version(&bundle)?;
    for component in [
        LtxComponent::TextEncoder,
        LtxComponent::AudioVae,
        LtxComponent::DurationHead,
        LtxComponent::SpatialUpsampler,
        LtxComponent::TemporalUpsampler,
    ] {
        bundle.require(component)?;
    }
    bundle.require(ltx25_video_component(spec))?;
    let memory_strategy = crate::memory_strategy_2_5::memory_strategy_contract(spec)?;
    let memory_tier = crate::memory_strategy_2_5::resolved_numeric_tier(spec)?;
    Ok(Box::new(Ltx25 {
        descriptor: descriptor_25_for_variant(variant),
        spec: spec.clone(),
        variant,
        memory_strategy,
        memory_tier,
        enhancer_cache: Rc::new(RefCell::new(PrefixCache::with_budget(
            LTX25_ENHANCER_PREFIX_CACHE_BYTES,
        ))),
    }))
}

/// Staging the diffusion-video-VAE component is the alternate-decoder selection contract.  A
/// bundle may contain both VAE variants for catalog discovery; only an explicit component choice
/// switches the provider away from its ordinary convolutional decoder.
pub(super) fn ltx25_video_component(spec: &LoadSpec) -> LtxComponent {
    match ltx25_decoder_selection(spec) {
        Ltx25DecoderSelection::Conv => LtxComponent::ConvVideoVae,
        Ltx25DecoderSelection::DiffusionBudgeted(_) => LtxComponent::DiffusionVideoVae,
    }
}

pub(super) fn ltx25_decoder_selection(spec: &LoadSpec) -> Ltx25DecoderSelection {
    if spec
        .components
        .contains_key(LtxComponent::DiffusionVideoVae.id())
    {
        Ltx25DecoderSelection::DiffusionBudgeted(DEFAULT_DIFFVAE_MODE)
    } else {
        Ltx25DecoderSelection::Conv
    }
}

/// Resolve the opt-in stock Gemma-4 enhancer below the one LTX-2.5 rehost.  The only accepted
/// override is another already-staged directory; no cache/environment/network lookup exists.
pub(crate) fn ltx25_enhancer_dir(spec: &LoadSpec) -> Result<PathBuf> {
    if let Some(source) = spec.components.get(LTX25_ENHANCER_COMPONENT) {
        return match source {
            WeightsSource::Dir(dir) => Ok(dir.clone()),
            WeightsSource::File(path) => Err(Error::Msg(format!(
                "ltx_2_5: '{LTX25_ENHANCER_COMPONENT}' must be the staged google/gemma-4-12B-it \
                 snapshot directory, not file {}",
                path.display()
            ))),
        };
    }
    match &spec.weights {
        WeightsSource::Dir(root) => Ok(root.join(LTX25_ENHANCER_COMPONENT)),
        WeightsSource::File(_) => Err(Error::Msg(
            "ltx_2_5: split bundles must be loaded from their component directory".into(),
        )),
    }
}

/// Converted LTX-2.5 tiers split each VAE encoder into its own file, while raw split bundles keep
/// the encoder beside its decoder. The bundle resolver owns component discovery; after it has
/// selected a component, prefer that component's documented tier encoder half when present.
pub(crate) fn ltx25_encoder_path(
    root: &std::path::Path,
    component: LtxComponent,
    fallback: &std::path::Path,
) -> PathBuf {
    let half = match component {
        LtxComponent::ConvVideoVae => root.join("vae_encoder.safetensors"),
        LtxComponent::DiffusionVideoVae => root.join("diffusion_vae_encoder.safetensors"),
        _ => return fallback.to_path_buf(),
    };
    if half.is_file() {
        half
    } else {
        fallback.to_path_buf()
    }
}

pub(super) fn build_ltx25(
    spec: &LoadSpec,
    enhancer_cache: Rc<RefCell<PrefixCache>>,
) -> Result<Ltx> {
    let bundle = crate::bundle::resolve_split_bundle(spec)?;
    let variant = crate::dev_sampler::from_bundle(&bundle)?;
    let config = LtxConfig::from_bundle(&bundle)?;
    let memory_strategy = crate::memory_strategy_2_5::memory_strategy_contract(spec)?;
    let memory_tier = crate::memory_strategy_2_5::resolved_numeric_tier(spec)?;
    crate::bundle::assert_gemma_version(&bundle)?;
    let root = match &spec.weights {
        WeightsSource::Dir(root) => root,
        WeightsSource::File(_) => {
            return Err(Error::Msg(
                "ltx_2_5: split bundles must be loaded from their component directory".into(),
            ))
        }
    };
    let split = SplitModel::from_model_dir(root)?;
    let (dit_prec, stat_dt) = match spec.precision {
        LoadPrecision::Bf16 => (
            Precision::quant_bf16(split.bits, split.group),
            Dtype::Bfloat16,
        ),
        LoadPrecision::Fp32 => (
            Precision::quant_f32(split.bits, split.group),
            Dtype::Float32,
        ),
    };
    let decoder_selection = ltx25_decoder_selection(spec);
    let video_component = ltx25_video_component(spec);
    let video_path = bundle.require(video_component)?.path().to_path_buf();
    let encoder_path = ltx25_encoder_path(root, video_component, &video_path);
    let audio = bundle.require(LtxComponent::AudioVae)?.path().to_path_buf();
    // The converted vocoder is an intentionally unclaimed secondary sibling, like the connector.
    // Resolve it beside the audio VAE that owns the merged upstream audio/vocoder configuration.
    let vocoder = audio.with_file_name("vocoder.safetensors");
    let connector = bundle
        .require(LtxComponent::Transformer)?
        .path()
        .with_file_name("connector.safetensors");
    // The converter keeps the connector beside the transformer.  A separately staged connector is
    // not a public component because the transformer metadata remains its identity authority.
    if !connector.exists() {
        return Err(Error::Msg(format!(
            "ltx_2_5: missing connector.safetensors beside transformer {}",
            bundle.require(LtxComponent::Transformer)?.path().display()
        )));
    }
    let video_w = Weights::from_file(&video_path)?;
    let audio_w = Weights::from_file(&audio)?;
    let vocoder_w = Weights::from_file(&vocoder)?;
    // Resident components cross the GPU-view guard here (sc-22414, `mlx_gen::coherence`).
    video_w.materialize()?;
    audio_w.materialize()?;
    vocoder_w.materialize()?;
    let vae_cfg = LtxVaeConfig::from_bundle(&bundle, video_component)?;
    let audio_cfg = AudioVaeConfig::from_bundle(&bundle)?;
    let vocoder_cfg = VocoderConfig::from_bundle(&bundle)?;
    let (vae, decoder) = match (video_component, decoder_selection) {
        (LtxComponent::ConvVideoVae, Ltx25DecoderSelection::Conv) => (
            LtxVideoVae::from_weights_lazy_encoder(&video_w, encoder_path, &vae_cfg)?,
            Ltx25Decoder::Conv,
        ),
        (LtxComponent::DiffusionVideoVae, Ltx25DecoderSelection::DiffusionBudgeted(mode)) => {
            let diff_cfg = NaDiffusionDecoderConfig::from_embedded_vae(
                bundle.require(LtxComponent::DiffusionVideoVae)?.config()?,
            )?;
            let quant = split.quantized.then_some(DiffVaeQuant {
                bits: split.bits,
                group: split.group,
            });
            let decoder = NaDiffusionDecoder::from_weights(&video_w, &diff_cfg, quant)?;
            // The diffusion VAE carries the same causal encoder used for image conditioning, but
            // no convolutional decoder.  Keeping it encoder-only prevents a staged DiffVAE route
            // from materialising and then accidentally decoding through the conv sibling.
            (
                LtxVideoVae::encoder_only_lazy(encoder_path, &vae_cfg)?,
                Ltx25Decoder::Diffusion {
                    decoder: Box::new(decoder),
                    mode,
                },
            )
        }
        _ => unreachable!("the staged LTX-2.5 decoder selection and component must agree"),
    };
    let audio_decoder = AudioDecoder::from_weights(&audio_w, &audio_cfg)?;
    let vocoder = LtxVocoder::from_weights(&vocoder_w, &vocoder_cfg)?;
    let spatial = bundle.require(LtxComponent::SpatialUpsampler)?.path();
    let upsampler = LatentUpsampler::from_checkpoint(spatial)?;
    let latent_mean = to_dtype(video_w.require("per_channel_statistics.mean")?, stat_dt)?;
    let latent_std = to_dtype(video_w.require("per_channel_statistics.std")?, stat_dt)?;
    let te_path = bundle
        .require(LtxComponent::TextEncoder)?
        .path()
        .to_path_buf();
    let assets = GemmaAssets::load(&te_path)?;
    let tokenizer = Ltx25Tokenizer::from_assets(&assets)?;
    let enhancer_template = ltx25_packed_chat_template(&assets)?;
    let enhancer = ltx25_enhancer_dir(spec)?;
    let duration_w = Weights::from_file(bundle.require(LtxComponent::DurationHead)?.path())?;
    duration_w.materialize()?;
    let duration_head = DurationHead::from_weights(&duration_w)?;
    let temporal_upsampler =
        LatentUpsampler::from_checkpoint(bundle.require(LtxComponent::TemporalUpsampler)?.path())?;

    Ok(Ltx {
        descriptor: descriptor_25_for_variant(variant),
        memory_strategy: Some(memory_strategy),
        memory_tier: Some(memory_tier),
        memory_overlay: None,
        tokenizer: Tokenizer::Gemma4(tokenizer),
        transformer_path: bundle
            .require(LtxComponent::Transformer)?
            .path()
            .to_path_buf(),
        config,
        dit_prec,
        adapters: spec.adapters.clone(),
        text_assets: TextAssets::Gemma4 {
            bundle,
            connector,
            offload_policy: spec.offload_policy,
            enhancer,
            enhancer_template,
        },
        execution: LtxExecution::Ltx25(Box::new(Ltx25Execution {
            variant,
            duration_head,
            temporal_upsampler,
            decoder,
        })),
        uncensored_enhancer: None,
        enhancer_cache: Some(enhancer_cache),
        upsampler,
        vae,
        audio_decoder,
        vocoder,
        latent_mean,
        latent_std,
        audio_sample_rate: vocoder_cfg.final_sample_rate() as u32,
        stat_dt,
    })
}
