//! Asset and architecture facts, and load-exact structural resident evidence.

use super::*;

/// Provider-owned, load-exact asset facts used as formula inputs.
///
/// `base_bytes` is exactly the sum of the three base-model component fields. Auxiliary networks
/// never belong in any of those four fields; they are declared once in `overlay_bytes` and, when
/// non-zero alongside both formula variables, by typed resident components.
///
/// ## What "auxiliary" means (SC-22667)
///
/// **Auxiliary means outside the base model, not merely outside the usual three names.** The set is
/// closed and enumerated by [`MemoryComponentKind::is_auxiliary`]: a control branch, an adapter
/// stack, an IP-Adapter, an identity encoder, a reference (VAE) encoder, a prompt upsampler — networks a *request* opts
/// into, which a plain render of the same checkpoint does not load. Those are the ones
/// `overlay_bytes` exists for, and the reason it is excluded from `base_bytes` is precisely that a
/// fit decision for a request without them must not carry them.
///
/// A base-model network the route **always** loads is therefore charged in one of the three base
/// fields even when it is none of "the text encoder", "the DiT" or "the image VAE" — LTX-2.5's
/// latent upsamplers, audio VAE, vocoder and duration head are all of this kind. Each goes in the
/// field naming the role it plays (a duration head conditions, an upsampler is part of the denoise
/// stack, an audio decoder decodes), charged exactly once, under the *One network, one field* rule
/// below. Moving them to `overlay_bytes` would be a byte-accounting **error**, not a stylistic
/// choice: [`MemoryProviderContract::total_resident_bytes`] adds `overlay_bytes` only for a
/// contract that also declares typed auxiliary components, so bytes parked there by a provider with
/// no such component simply vanish from the total.
///
/// ## One network, one field
///
/// Every base-model network is charged in exactly **one** of the three component fields, even when
/// the provider runs it in more than one [`MemoryPhase`]. The canonical case is an edit or img2img
/// route that encodes its reference through the same autoencoder weights it later decodes with:
/// those bytes belong to `decoder_bytes` alone, and `conditioning_bytes` stays the text/vision
/// encoder stack. Folding the VAE into `conditioning_bytes` as well would either break
/// `base_bytes == sum` or, if `base_bytes` were computed from the folded sum, charge one resident
/// network twice against every fit decision.
///
/// The autoencoder's two halves are **not** one network for this rule (SC-22667). A text-to-image
/// render decodes and never encodes, so a loader that builds the `encoder.*` half only when a
/// reference arrives declares that half as a [`MemoryComponentKind::ReferenceEncoder`] auxiliary
/// component in `overlay_bytes`, and `decoder_bytes` holds only what the plain render keeps
/// resident. Pricing the whole `vae/` file into `decoder_bytes` put the MLX Z-Image resident sum
/// 58.6 MB *above* the measured resident level of a text-to-image anchor, which the fit domain
/// guard rightly refused.
///
/// The phase in which a shared network is *resident* is a lifecycle property, not a byte-count one.
/// The contract has no per-phase residency declaration for base components today — only the
/// typed auxiliary components in [`MemoryFormulaKind::ComponentPhaseEnvelope`] carry a
/// [`MemoryComponentResidency`] — so a provider whose decoder is co-resident during conditioning
/// documents that on its own contract builder and leaves these fields as the one-network-one-field
/// decomposition.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemoryAssetFacts {
    pub base_bytes: u64,
    /// The conditioning encoder stack (text and/or vision encoders). A shared autoencoder used as
    /// the reference encoder is **not** included here; see *One network, one field* above.
    pub conditioning_bytes: u64,
    pub transformer_bytes: u64,
    /// The autoencoder weights a plain render keeps resident, charged once whether they only
    /// decode or also encode a reference during conditioning. An encoder half the loader
    /// materializes only for a reference request is a [`MemoryComponentKind::ReferenceEncoder`]
    /// auxiliary component instead.
    pub decoder_bytes: u64,
    /// Compatibility aggregate for auxiliary resident networks. An adopting provider declares the
    /// corresponding typed contributions in [`MemoryFormulaKind::ComponentPhaseEnvelope`].
    pub overlay_bytes: u64,
}

/// Provider-owned model architecture facts used as activation-formula inputs.
///
/// An axis is *derived from a component config*: parsed out of the materialized snapshot by a
/// provider whose loader parses it, or mirrored as a crate constant by a provider whose loader
/// builds the same geometry in code. Which of the two a provider may use is a property of its
/// backend, checked by the registry gate rather than expressible in this type.
///
/// Every axis is an `Option` on purpose. `None` means **not declared**, and a provider must use it
/// for two distinct cases which both differ from `Some(0)`:
///
/// * the axis is *structurally absent* for this architecture (an image model has no temporal VAE
///   scale), and
/// * the provider could not read the axis (a weights-free contract built before the snapshot exists
///   on disk).
///
/// A zero is never a legitimate value for any of these axes, so a defaulted `0` would silently
/// poison any activation estimate that multiplied by it. Providers therefore derive these from the
/// component config their own loader uses - the snapshot's `config.json` where it parses one, the
/// crate's own model-config constructor where it builds the geometry in code - and publish `None`
/// otherwise.
///
/// # The preset-fallback rule (SC-22667)
///
/// **Fall back to a preset only where the LOADER itself falls back to that preset; otherwise declare
/// the axis `None`.** A contract that substitutes a crate preset for an axis the loader would have
/// refused to load without is describing a render that cannot happen, and one that publishes `None`
/// for an axis the loader supplies from a preset is withholding a fact the run will use.
///
/// The rule is per axis and per parse outcome, not per provider, because both backends' config
/// readers mix the two behaviours:
///
/// * A reader that defaults **per key** (candle Krea's `Krea2Config::from_json` takes each missing
///   key from `Krea2Config::turbo`) makes the preset the loaded value, so reading the axes back off
///   the parsed struct is correct — including for keys the file omitted.
/// * A reader that **requires** every key (candle Mage's `MageConfig::from_json`, MLX Mage's
///   `MageFlowConfig::from_transformer_config_json`) fails the load on a partial file, so a partial
///   file has no loadable geometry and every axis it would have supplied is `None`. Substituting
///   the crate preset there — `.ok().unwrap_or_else(preset)` — publishes numbers no load will use.
/// * An axis the loader **never** reads from a config, because it constructs that component from
///   code (Mage's CoD decoder, Krea's reused Qwen-Image VAE), is a preset axis unconditionally: the
///   crate constant *is* what the loader builds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemoryArchitectureFacts {
    /// Attention heads in one transformer block.
    pub attention_heads: Option<u32>,
    /// Per-head channel width (`hidden_dim / attention_heads` for a uniform-head architecture).
    pub head_dim: Option<u32>,
    /// Number of transformer blocks the denoiser stacks.
    pub transformer_blocks: Option<u32>,
    /// Spatial patchification factor applied to the latent before the transformer.
    pub patch_size: Option<u32>,
    /// Latent channel count produced by the encoder / consumed by the decoder.
    pub latent_channels: Option<u32>,
    /// Pixels per latent unit on each spatial axis (8 for a four-stage image VAE).
    pub vae_spatial_scale: Option<u32>,
    /// Frames per latent unit on the temporal axis. Structurally absent — and therefore `None` —
    /// for image models.
    pub vae_temporal_scale: Option<u32>,
    /// Bytes per element of the activation dtype (2 for bf16/f16, 4 for f32).
    ///
    /// **Which dtype, on a provider whose phases differ (SC-22667):** the width the **denoise**
    /// phase computes in — the transformer's activation dtype. Almost every provider here runs its
    /// DiT bf16 while opening the autoencoder f32, and some upcast conditioning per matmul, so
    /// "the" activation width is only well defined once it is pinned to one phase.
    ///
    /// Denoise is the phase this names because it is the phase this axis is *for*: the other seven
    /// axes are attention and latent geometry, and the estimate they feed multiplies out the
    /// attention scores and per-block activations of the trunk. A VAE width published here would be
    /// multiplied into an attention formula that never runs at that width.
    ///
    /// It follows that this axis is **not** a per-component byte fact and must never be used to
    /// price one: the bytes a decoder or encoder materializes belong in
    /// [`MemoryAssetFacts::decoder_bytes`] / [`MemoryAssetFacts::conditioning_bytes`], measured at
    /// the width that component is actually opened in.
    pub activation_dtype_width: Option<u32>,
}

impl MemoryArchitectureFacts {
    /// Whether no axis at all has been declared. A contract publishing a lifecycle-phase formula
    /// with nothing declared cannot support an activation-aware estimate.
    pub const fn is_empty(&self) -> bool {
        self.attention_heads.is_none()
            && self.head_dim.is_none()
            && self.transformer_blocks.is_none()
            && self.patch_size.is_none()
            && self.latent_channels.is_none()
            && self.vae_spatial_scale.is_none()
            && self.vae_temporal_scale.is_none()
            && self.activation_dtype_width.is_none()
    }

    /// Whether at least one axis derived from a component config has been declared - whether that
    /// config was parsed from the snapshot or mirrored as a crate constant.
    ///
    /// Both derivations count on purpose. A Candle provider that parses `transformer/config.json`
    /// and an MLX provider that mirrors the same geometry as compile-time preset constants have
    /// each stated the architecture the loader will build; only the *source* differs, and every
    /// caller of this predicate cares that the geometry was stated at all. Which of the two a given
    /// surface is *allowed* to use is a separate, backend-keyed rule that lives on the registry
    /// gate, not here.
    ///
    /// [`Self::activation_dtype_width`] is deliberately excluded: an adopting provider emits it
    /// from a compile-time dtype constant that is identical for every model the crate loads, so it
    /// is `Some` whenever the provider was compiled - never evidence that any component geometry
    /// was stated. A gate that accepts it as "at least one architecture fact" is satisfied by a
    /// contract that declared nothing about the model.
    pub const fn has_declared_architecture_axis(&self) -> bool {
        self.attention_heads.is_some()
            || self.head_dim.is_some()
            || self.transformer_blocks.is_some()
            || self.patch_size.is_some()
            || self.latent_channels.is_some()
            || self.vae_spatial_scale.is_some()
            || self.vae_temporal_scale.is_some()
    }

    /// Axes declared as `Some(0)`. Zero is never legitimate for any axis: an absent axis is `None`.
    pub fn zero_valued_axes(&self) -> Vec<&'static str> {
        [
            ("attention_heads", self.attention_heads),
            ("head_dim", self.head_dim),
            ("transformer_blocks", self.transformer_blocks),
            ("patch_size", self.patch_size),
            ("latent_channels", self.latent_channels),
            ("vae_spatial_scale", self.vae_spatial_scale),
            ("vae_temporal_scale", self.vae_temporal_scale),
            ("activation_dtype_width", self.activation_dtype_width),
        ]
        .into_iter()
        .filter_map(|(name, value)| (value == Some(0)).then_some(name))
        .collect()
    }
}

/// Immutable provider-owned facts for a narrowly pinned Resident route.
///
/// This is intentionally separate from [`MemoryCalibrationIdentity`]: it proves what was loaded and
/// how many bytes it contains, but makes no measured-peak claim.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MemoryStructuralResidentEvidence {
    pub abi: u32,
    pub provider_id: String,
    pub repository: String,
    pub revision: String,
    pub variant: String,
    pub receipt_sha256: String,
    pub tier: MemoryNumericTier,
    pub load_shape: LoadShape,
    pub asset_facts: MemoryAssetFacts,
    /// Request-scoped bytes needed while adapters are folded. These are not steady overlay bytes.
    pub request_transient_bytes: u64,
    pub direct_file_count: u32,
    pub adapter_count: u32,
}

/// Exact execution identity paired with [`MemoryStructuralResidentEvidence`].
///
/// `source_digest` is an opaque 64-hex digest produced by the caller, so this contract does not
/// depend on SceneWorks assets or on any particular notion of a "source id". Both SCAIL-2 providers
/// derive it from the **carrier bytes** of the request (the character image, mask and driving
/// frames) rather than from an asset identifier; a different provider is free to digest whatever
/// makes its execution identity exact.
#[derive(Clone, Debug, PartialEq)]
pub struct MemoryStructuralResidentRequestIdentity {
    pub source_digest: String,
    pub mode: String,
    pub carrier_shape: String,
    pub width: u32,
    pub height: u32,
    pub frames: u32,
    pub fps: u32,
    pub reference_count: u32,
    pub seed: Option<u64>,
    pub sampler: Option<String>,
    pub scheduler: Option<String>,
    pub steps: Option<u32>,
    pub guidance: Option<f32>,
    pub scheduler_shift: Option<f32>,
    pub selection: MemorySelection,
}

impl MemoryStructuralResidentEvidence {
    pub fn validate(&self) -> Result<()> {
        if self.abi != MEMORY_STRUCTURAL_RESIDENT_EVIDENCE_ABI
            || self.provider_id.is_empty()
            || self.repository.is_empty()
            || self.revision.is_empty()
            || self.variant.is_empty()
            || self.receipt_sha256.len() != 64
            || !self
                .receipt_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
            || self.asset_facts.base_bytes == 0
            || self.direct_file_count == 0
        {
            return Err(Error::Unsupported(
                "invalid structural Resident evidence".to_owned(),
            ));
        }
        Ok(())
    }

    /// Domain prefix of SCAIL-2's structural Resident receipt.
    ///
    /// The domain is a **caller-supplied** parameter of
    /// [`evidence_revision_in_domain`](Self::evidence_revision_in_domain), not a property of this
    /// generic contract: the contract itself knows nothing about SCAIL-2. This constant names the
    /// value SCAIL-2 has always used so the two provider crates keep parsing `parts[0]` unchanged,
    /// and so a second adopter is forced to mint its own domain rather than share this one.
    pub const SCAIL2_RESIDENT_EVIDENCE_DOMAIN: &'static str = "scail2-resident-v1";

    /// SCAIL-2's structural Resident receipt — [`Self::evidence_revision_in_domain`] at
    /// [`Self::SCAIL2_RESIDENT_EVIDENCE_DOMAIN`].
    pub fn evidence_revision(
        &self,
        request: &MemoryStructuralResidentRequestIdentity,
    ) -> Result<String> {
        self.evidence_revision_in_domain(Self::SCAIL2_RESIDENT_EVIDENCE_DOMAIN, request)
    }

    /// Structural Resident receipt for one execution identity, namespaced to `domain`.
    ///
    /// `domain` is both the plain-text prefix and a hashed input, so two providers that share this
    /// contract can never mint the same digest for the same request identity.
    ///
    /// Every `Option` knob is **tag-hashed**: a discriminant byte (`0` absent / `1` present)
    /// precedes the value bytes. Hashing `unwrap_or_default()` instead would collapse `None` onto
    /// `Some(0)`, `Some(0.0)` and `Some("")` — the same absent-versus-zero collapse the Wan video
    /// receipt is written to avoid — which would let a request that dropped a knob reuse the
    /// receipt of a request that set it to zero.
    pub fn evidence_revision_in_domain(
        &self,
        domain: &str,
        request: &MemoryStructuralResidentRequestIdentity,
    ) -> Result<String> {
        self.validate()?;
        if domain.is_empty() || domain.contains(':') {
            return Err(Error::Unsupported(
                "structural Resident evidence domain must be a non-empty colon-free token"
                    .to_owned(),
            ));
        }
        if request.source_digest.len() != 64
            || !request
                .source_digest
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(Error::Unsupported(
                "invalid structural Resident source identity".to_owned(),
            ));
        }
        let mut hasher = Sha256::new();
        for value in [
            domain,
            self.provider_id.as_str(),
            self.repository.as_str(),
            self.revision.as_str(),
            self.receipt_sha256.as_str(),
            request.source_digest.as_str(),
            request.mode.as_str(),
            request.carrier_shape.as_str(),
        ] {
            hasher.update(value.as_bytes());
            hasher.update([0]);
        }
        // `variant` stays a distinct field rather than joining the loop above: it is part of the
        // artifact identity, and the loop's `0` separator already keeps neighbours from sliding
        // into one another.
        hasher.update(self.variant.as_bytes());
        hasher.update([0]);
        for value in [request.sampler.as_deref(), request.scheduler.as_deref()] {
            hasher.update([u8::from(value.is_some())]);
            hasher.update(value.unwrap_or_default().as_bytes());
            hasher.update([0]);
        }
        hasher.update(request.width.to_le_bytes());
        hasher.update(request.height.to_le_bytes());
        hasher.update(request.frames.to_le_bytes());
        hasher.update(request.fps.to_le_bytes());
        hasher.update(request.reference_count.to_le_bytes());
        hasher.update([u8::from(request.seed.is_some())]);
        hasher.update(request.seed.unwrap_or_default().to_le_bytes());
        hasher.update([u8::from(request.steps.is_some())]);
        hasher.update(request.steps.unwrap_or_default().to_le_bytes());
        for value in [request.guidance, request.scheduler_shift] {
            hasher.update([u8::from(value.is_some())]);
            hasher.update(value.unwrap_or_default().to_bits().to_le_bytes());
        }
        hasher.update(format!("{:?}", request.selection));
        Ok(format!(
            "{domain}:{}:{}:{:x}",
            self.receipt_sha256,
            request.source_digest,
            hasher.finalize()
        ))
    }
}
