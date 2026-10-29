//! `LoadSpec`: the provider load request and its validation.

use super::*;

/// How to load a model. `weights` is required; everything else defaults to dense bf16. The
/// device is the process-default Metal GPU — the crate runs single-device (the MLX default
/// device is not thread-safe; the worker serializes jobs per thread).
#[derive(Clone, Debug)]
pub struct LoadSpec {
    pub weights: WeightsSource,
    /// Exact caller-resolved catalog route served by a shared provider. `None` preserves the
    /// historical provider-id identity; callers that route several catalog entries through one
    /// engine (for example the five SDXL entries) set this so route-specific quality evidence can
    /// never authorize a sibling checkpoint.
    pub resolved_route: Option<String>,
    /// Sealed production-latent quality rows packaged by the caller for this exact loaded route.
    /// Empty preserves every provider's historical bounded-decode declaration. Providers accept
    /// rows only when their route/backend/numeric-tier contract validates them; the semantic records
    /// are independent from model-memory calibration evidence.
    pub decode_geometry_policies: Vec<MemoryDecodeGeometryPolicy>,
    /// True once a caller declares quality policy for this route, even if exact tier/load-shape
    /// projection or an independently audited binding leaves zero usable rows. Providers use this
    /// to distinguish fail-closed semantic scope from a genuinely legacy empty table.
    pub decode_geometry_policy_authoritative: bool,
    /// Independently resolved runtime identity for the packaged quality rows. A non-empty table
    /// without this binding is malformed and fails closed before provider construction.
    pub decode_quality_runtime_identity: Option<MemoryDecodeQualityRuntimeIdentity>,
    /// Caller-prepared identities for every single-file source carried by this spec.
    ///
    /// SceneWorks resolves and confines primary weights, controls, adapters, typed auxiliaries, and
    /// named components before it derives cache identity. Keeping those exact
    /// [`PinnedWeightsFile`] tokens on the load spec lets the cache key and provider consume one
    /// identity per lexical path instead of independently re-pinning whatever occupies that path at
    /// two different times. Keys are absolute but otherwise lexical, so an extension-bearing HF
    /// snapshot link is never canonicalized to its extensionless blob target.
    ///
    /// [`PreparedFilePins`] carries an explicit mode bit: an ordinary caller stays in compatibility
    /// mode and [`Self::file_pin_for`] pins the requested source on demand, while an intentionally
    /// prepared spec remains prepared even when it has no File sources. In prepared mode every
    /// configured File source must have a matching token; a missing or orphaned token fails closed.
    /// Prefer [`Self::prepare_file_sources`] or [`Self::with_prepared_file_pin`] to enter that mode.
    pub(super) prepared_file_pins: PreparedFilePins,
    /// Additional exact files retained by a validated component receipt. These paths are not
    /// ordinary provider slots: a single-file encoder's sibling config and a complete snapshot's
    /// tokenizer are behavior-bearing inputs even though the backend only receives the encoder
    /// source. Only crate-owned validation APIs may populate this set.
    pub(super) prepared_receipt_paths: BTreeSet<PathBuf>,
    /// Exact selected-encoder source shape and direct-shard inventory paired with
    /// [`prepared_receipt_paths`](Self::prepared_receipt_paths). Known file tokens reject mutation;
    /// this receipt additionally rejects directory additions, removals, renames, and type changes.
    pub(super) prepared_encoder_receipt:
        Option<crate::encoder_contract::PreparedEncoderLoadReceipt>,
    pub(super) source_inventory: Option<std::sync::Arc<crate::SourceInventory>>,
    pub quantize: Option<Quant>,
    pub precision: Precision,
    /// YuE2-only AR execution mode; every other provider refuses a non-native value.
    pub yue2_ar_mode: Yue2ArMode,
    /// YuE2-only stage compute policy; other providers refuse a non-legacy value.
    pub yue2_compute_policy: Yue2ComputePolicy,
    /// Auxiliary control-branch weights overlaid onto the base model at load time — a ControlNet
    /// checkpoint applied on top of `weights` (e.g. Z-Image's Fun-Controlnet-Union safetensors).
    /// `None` for the plain base model; a control-variant loader requires it. A load-time model
    /// *component* (it alters the graph), distinct from [`adapters`](Self::adapters) below, which
    /// are forward-time residual overlays on existing linears.
    pub control: Option<WeightsSource>,
    /// **Additional** ControlNet checkpoints for MultiControlNet (sc-3378) — used by providers that
    /// sum several control branches (the SDXL provider). These are loaded *after* [`control`](Self::control)
    /// and paired, in order, with the request's `Conditioning::Control` images (the diffusers
    /// `MultiControlNetModel` order semantics: branch *i* ← the *i*-th `Control`). Empty for the
    /// single-branch case (then only `control` is used); providers that do not support multi-control
    /// (Z-Image / Qwen union checkpoints) ignore this field.
    pub extra_controls: Vec<WeightsSource>,
    /// Auxiliary **IP-Adapter** weights overlaid at load time (sc-3059) — the image-prompt
    /// conditioning checkpoint (image encoder + Resampler + decoupled cross-attn K/V), e.g. an
    /// `h94/IP-Adapter`-layout snapshot dir. `None` for the plain base model. Like
    /// [`control`](Self::control), a load-time graph *component* (it adds K/V projections to the
    /// cross-attention), distinct from forward-time [`adapters`](Self::adapters).
    pub ip_adapter: Option<WeightsSource>,
    /// LoRA/LoKr adapters baked onto the model at load time. Multiples + mixed LoRA/LoKr stack by
    /// construction (see the provider `adapters` modules). Applied during `load` on the still-mutable
    /// model — the seam, since `Generator::generate`/`Transform::apply` take `&self` and the frozen
    /// fork likewise applies adapters in its initializer. Changing the adapter set means reloading.
    pub adapters: Vec<AdapterSpec>,
    /// Auxiliary **PiD** (NVIDIA Pixel-Diffusion) decoder weights overlaid at load time (epic 7840) —
    /// the optional super-resolving replacement for the model's VAE decode step. `None` for the plain
    /// VAE-decoding model; `Some` makes the PiD decoder *available*, after which the per-generation
    /// [`crate::GenerationRequest::use_pid`] flag selects it at the decode call site. Like
    /// [`control`](Self::control)/[`ip_adapter`](Self::ip_adapter) it is a load-time component (PiD's
    /// net + Gemma-2 caption encoder are heavy, so they load once and the toggle rides each request);
    /// only providers whose latent space has a PiD backbone read it (Qwen-Image / Krea today —
    /// sc-7845), and they ignore it when the request does not request PiD.
    pub pid: Option<PidWeights>,
    /// Auxiliary **identity-conditioning** sub-model weights (PuLID / InstantID family, sc-8827) — the
    /// EVA-CLIP tower, the identity encoder checkpoint, and the native face-analysis weight dir that a
    /// face-ID provider needs on top of its diffusion backbone. `None` for a plain base model. A
    /// face-ID provider that reads this slot **requires** it: the caller drives every identity path
    /// through the spec (backend-neutral — just paths), and an absent slot (or an absent sub-field) is a
    /// **load-time** error, never a fetch from an env var or a derived on-disk cache (epic 13657, sc-13664 —
    /// the PuLID-FLUX loader dropped its historical `PULID_*` env / HF-cache-derived fallbacks).
    pub identity: Option<IdentityWeights>,
    /// Auxiliary **external text-encoder** snapshot directory (sc-8827) — a separate TE snapshot a
    /// provider loads alongside its main checkpoint, e.g. LTX-2.3's Gemma-3-12B encoder (which is not
    /// bundled in the checkpoint dir). A provider that reads this slot **requires** it: the caller
    /// drives the TE location through the spec (backend-neutral — just a path), and an absent slot is a
    /// **load-time** error, never a process-global env var or a derived on-disk cache scan (epic 13657,
    /// sc-13664 — LTX-2.3 dropped its historical `$LTX_GEMMA_DIR` / HF-cache-derived fallbacks).
    pub text_encoder: Option<WeightsSource>,
    /// Component-residency strategy (epic 10765, sc-10821). [`OffloadPolicy::Resident`] (default) keeps
    /// every component resident for the whole generation; [`OffloadPolicy::Sequential`] asks a supporting
    /// provider to load→use→drop each heavy component after its phase so peak VRAM is the largest single
    /// working set, not the sum. Advisory — a provider that has not wired the residency lifecycle
    /// ignores it and stays `Resident`; [`Capabilities::supports_sequential_offload`](crate::generator::Capabilities::supports_sequential_offload)
    /// advertises which engines honor it (sc-11126). Backend-neutral.
    pub offload_policy: OffloadPolicy,
    /// Weight materialization shape, independent from [`offload_policy`](Self::offload_policy).
    /// The default preserves the historical eager/warm path. A deferred shape is meaningful only
    /// for providers with a re-openable source such as a snapshot directory.
    pub load_shape: LoadShape,
    /// Caller-owned declaration authority for [`load_shape`](Self::load_shape). This participates
    /// in request-scoped execution-policy identity whenever orchestration reconstructs a
    /// [`LoadSpec`], so an authoritative refusal cannot alias a not-yet-evaluated eager load. It is
    /// not immutable weight identity: residency and materialization policy do not change the model
    /// bytes.
    pub load_shape_declaration_result: LoadShapeDeclarationResult,
    /// **Named, caller-provisioned model components** (epic 13657) — the generic, additive home for
    /// the extra weight artifacts a model needs beyond its base `weights` and the typed overlays
    /// above, keyed by a stable component id. The complement of
    /// [`ModelDescriptor::required_components`](crate::generator::ModelDescriptor::required_components):
    /// the descriptor *advertises* which ids a model requires (weights-free, so a consumer knows what
    /// to stage), and this map *carries* the resolved local path for each. A provider reads each id at
    /// load time via [`require_component`](crate::control::require_component); a required id absent
    /// here is a **load-time** contract error, not a mid-render fetch (the whole point of the seam —
    /// it converts e.g. perth's mid-render watermark-weight fetch into a load contract error), and an
    /// unrecognized id is rejected via
    /// [`reject_unknown_components`](crate::control::reject_unknown_components). Default empty; set
    /// with [`with_component`](Self::with_component), mirroring [`with_control`](Self::with_control).
    ///
    /// This is deliberately a `BTreeMap<String, WeightsSource>` (not a typed slot per component and
    /// not a new [`WeightsSource`] hub-fetch variant — both alternatives were rejected in the sc-13591
    /// research): components are model-specific and open-ended, so a generic keyed map lets a new
    /// model declare new ids without a contract edit, while the descriptor's `required_components`
    /// keeps the set discoverable and conformance-checked.
    ///
    /// ## Provider → component-id registry (the reserved ids downstream stories consume)
    ///
    /// This map is the registry of record for component ids. The provisional set (epic 13657):
    ///
    /// | Model | Component ids |
    /// |-------|---------------|
    /// | chatterbox (TTS) | `perth`, `voice_embedding` |
    /// | MOSS tts / tts-realtime | `codec` |
    /// | SDXL | `tokenizer_clip_l`, `tokenizer_clip_bigg`, `vae_fp16_fix` |
    /// | mmaudio | `clip`, `synchformer`, `dit`, `vae`, `vocoder` |
    /// | sensenova (fast) | `distill_lora` |
    /// | LTX-2.3 | `uncensored_enhancer` |
    /// | acestep (Cover) | `sft_cover` |
    /// | Iris-3B (generation) | `text_encoder` |
    ///
    /// sc-13664 wired sensenova's `distill_lora` (the 8-step distill LoRA for `sensenova_u1_8b_fast`,
    /// with a co-located-in-snapshot fallback; **not** a universally-`required_components` id, because a
    /// pre-merged turnkey tier bakes the merge in and needs no LoRA) and LTX-2.3's optional
    /// `uncensored_enhancer` (the amoral 4-bit Gemma enhancer, read on demand when a request sets
    /// `use_uncensored_enhancer`). acestep's `sft_cover` follows the same on-demand shape: the ~7.8 GB
    /// sft Cover snapshot dir, read only for a `Cover` audio-edit request, so it is likewise **not** a
    /// `required_components` id — text2music and the region edit modes load without it. LTX-2.3's
    /// *main* Gemma text encoder rides the typed [`text_encoder`](Self::text_encoder) slot (now
    /// required), not this map. Ids are lowercase
    /// `snake_case` registry identifiers (same shape as a descriptor `id`); a model's declared
    /// `required_components` ids are validated non-empty and unique by the descriptor conformance sweep
    /// ([`model_descriptor_errors`](crate::registry::model_descriptor_errors)).
    pub components: BTreeMap<String, WeightsSource>,
}

/// Where the optional PiD decoder's weights come from (epic 7840). A PiD decoder is tied to a
/// *latent space*, not a model, so a provider in an eligible space points at the converted
/// per-latent-space checkpoint plus the shared Gemma-2-2b caption encoder. Backend-neutral (just
/// paths); the tensor load lives in `mlx-gen-pid`.
#[derive(Clone, Debug)]
pub struct PidWeights {
    /// The converted PiD student checkpoint — a single `.safetensors`
    /// ([`WeightsSource::File`]; `tools/convert_pid.py` output for this latent space).
    pub checkpoint: WeightsSource,
    /// The `gemma-2-2b-it` snapshot **directory** ([`WeightsSource::Dir`]) — the caption encoder PiD
    /// conditions on (must contain the weights + `tokenizer.json`).
    pub gemma: WeightsSource,
}

/// The identity-conditioning sub-model weights a face-ID provider (PuLID / InstantID family) needs on
/// top of its diffusion backbone (F-114). Backend-neutral paths; the tensor load lives in the provider
/// crate.
///
/// Each field is `Option` only so the struct can be built incrementally / defaulted; a provider that
/// reads a field **requires** it — the caller supplies every path through this struct, and an absent
/// field is a **load-time** error (epic 13657, sc-13664). There is no env-var or cache fallback:
/// the old "optional field ⇒ provider `PULID_*` env / HF-cache-derived default" convention was
/// deleted, so a `None` a provider needs fails fast at load rather than silently scanning the disk.
#[derive(Clone, Debug, Default)]
pub struct IdentityWeights {
    /// The identity-encoder checkpoint — a single `.safetensors` (PuLID's
    /// `pulid_flux_v0.9.1.safetensors`). Required by the PuLID-FLUX loader (`None` ⇒ load-time error).
    pub encoder: Option<WeightsSource>,
    /// The converted EVA-CLIP vision tower — a single `.safetensors`. Required by the PuLID-FLUX loader
    /// (`None` ⇒ load-time error).
    pub eva: Option<WeightsSource>,
    /// The native face-analysis weight **directory** ([`WeightsSource::Dir`]) — must contain
    /// `scrfd_10g` / `arcface_iresnet100` / `bisenet_parsing` safetensors. Required by the PuLID-FLUX
    /// loader (`None` ⇒ load-time error).
    pub face_dir: Option<WeightsSource>,
}

impl LoadSpec {
    /// Full borrowed-spec ownership, guarded BEFORE the existing Clone executes.
    /// The exhaustive owned-field/container law covers every field, including
    /// nested quality policies, prepared receipts and split/deep component maps.
    /// SourceInventory Arcs remain shared; no image payload carrier is cloned.
    /// This owning law is Apple-qualified; other targets retain the explicit
    /// source-gap refusal, never an unbounded owning-Clone fallback.
    pub fn clone_with_capacity(
        &self,
        next: &mut dyn FnMut(u64) -> crate::Result<()>,
    ) -> crate::Result<Self> {
        #[cfg(target_vendor = "apple")]
        {
            use crate::source_inventory::bounded_clone::CloneLaw;
            next(self.clone_bytes()?)?;
            Ok(self.clone())
        }
        #[cfg(not(target_vendor = "apple"))]
        crate::SourceInventory::metadata_gap(next, crate::SourceInventory::PIN_CAPTURE_METADATA_GAP)
    }

    /// Dense bf16 load from the given source.
    pub fn new(weights: WeightsSource) -> Self {
        Self {
            weights,
            resolved_route: None,
            decode_geometry_policies: Vec::new(),
            decode_geometry_policy_authoritative: false,
            decode_quality_runtime_identity: None,
            prepared_file_pins: PreparedFilePins::default(),
            prepared_receipt_paths: BTreeSet::new(),
            prepared_encoder_receipt: None,
            source_inventory: None,
            quantize: None,
            precision: Precision::Bf16,
            yue2_ar_mode: Yue2ArMode::Native,
            yue2_compute_policy: Yue2ComputePolicy::Legacy,
            control: None,
            extra_controls: Vec::new(),
            ip_adapter: None,
            adapters: Vec::new(),
            pid: None,
            identity: None,
            text_encoder: None,
            offload_policy: OffloadPolicy::Resident,
            load_shape: LoadShape::EagerMaterialization,
            load_shape_declaration_result: LoadShapeDeclarationResult::NotEvaluated,
            components: BTreeMap::new(),
        }
    }

    /// Bind this load to the exact caller-resolved catalog route.
    pub fn with_resolved_route(mut self, resolved_route: impl Into<String>) -> Self {
        self.resolved_route = Some(resolved_route.into());
        self
    }

    /// Attach caller-packaged, sealed decode-quality rows for this load.
    pub fn with_decode_geometry_policies(
        mut self,
        policies: Vec<MemoryDecodeGeometryPolicy>,
    ) -> Self {
        self.decode_geometry_policy_authoritative |= !policies.is_empty();
        self.decode_geometry_policies = policies;
        self
    }

    /// Preserve a caller-audited semantic scope when its rows were stripped after a typed binding
    /// refusal. Setting this false never erases authority already established by attached rows.
    pub fn with_decode_geometry_policy_authority(mut self, authoritative: bool) -> Self {
        self.decode_geometry_policy_authoritative |= authoritative;
        self
    }

    /// Bind packaged decode-quality rows to the artifact actually being loaded.
    pub fn with_decode_quality_runtime_identity(
        mut self,
        identity: MemoryDecodeQualityRuntimeIdentity,
    ) -> Self {
        self.decode_quality_runtime_identity = Some(identity);
        self
    }

    /// Return only a route-bound quality table. A shared engine must never infer the catalog route
    /// from its provider id when semantic decode evidence is present.
    pub fn route_bound_decode_geometry_policies(
        &self,
    ) -> crate::Result<Vec<MemoryDecodeGeometryPolicy>> {
        if self.decode_geometry_policies.is_empty() {
            return Ok(Vec::new());
        }
        let policies = self.route_and_family_bound_decode_geometry_policies()?;
        self.validate_decode_quality_runtime_identity(&policies)?;
        Ok(policies)
    }

    fn route_and_family_bound_decode_geometry_policies(
        &self,
    ) -> crate::Result<Vec<MemoryDecodeGeometryPolicy>> {
        self.decode_quality_runtime_identity
            .as_ref()
            .ok_or_else(|| {
                crate::Error::Unsupported(
                    "decode-quality policies require an independently resolved artifact identity"
                        .to_owned(),
                )
            })?;
        let route = self.resolved_route.as_deref().ok_or_else(|| {
            crate::Error::Unsupported(
                "decode-quality policies require an exact caller-resolved route".to_owned(),
            )
        })?;
        if route.trim().is_empty() {
            return Err(crate::Error::Unsupported(
                "decode-quality policies require a non-empty caller-resolved route".to_owned(),
            ));
        }
        if let Some(foreign) = self
            .decode_geometry_policies
            .iter()
            .find(|policy| policy.resolved_route != route)
        {
            return Err(crate::Error::Unsupported(format!(
                "decode-quality policy route {:?} cannot authorize loaded route {:?}",
                foreign.resolved_route, route
            )));
        }
        if let Some(first) = self.decode_geometry_policies.first() {
            if let Some(foreign) = self
                .decode_geometry_policies
                .iter()
                .find(|policy| policy.family != first.family)
            {
                return Err(crate::Error::Unsupported(format!(
                    "decode-quality table for route {:?} cannot mix families {:?} and {:?}",
                    route, first.family, foreign.family
                )));
            }
        }
        Ok(self.decode_geometry_policies.clone())
    }

    fn validate_decode_quality_runtime_identity(
        &self,
        policies: &[MemoryDecodeGeometryPolicy],
    ) -> crate::Result<()> {
        let identity = self
            .decode_quality_runtime_identity
            .as_ref()
            .ok_or_else(|| {
                crate::Error::Unsupported(
                    "decode-quality policies require an independently resolved artifact identity"
                        .to_owned(),
                )
            })?;
        if let Some(foreign) = policies
            .iter()
            .find(|policy| policy.artifact != identity.artifact)
        {
            return Err(crate::Error::Unsupported(format!(
                "decode-quality policy artifact identity {:?} cannot authorize loaded identity {:?}",
                foreign.artifact, identity.artifact,
            )));
        }
        Ok(())
    }

    /// Select the rows applicable to one concrete loaded provider contract. A manifest table may
    /// contain several shipped numeric tiers and load shapes for the same catalog route; only the
    /// exact loaded tier and shape enter the runtime contract. A different backend is a malformed
    /// table, not an unrelated row to ignore.
    pub fn decode_geometry_policies_for_loaded_contract(
        &self,
        backend: MemoryBackend,
        tier: MemoryNumericTier,
    ) -> crate::Result<Vec<MemoryDecodeGeometryPolicy>> {
        if self.decode_geometry_policies.is_empty() {
            return Ok(Vec::new());
        }
        let policies = self.route_and_family_bound_decode_geometry_policies()?;
        if let Some(foreign) = policies.iter().find(|policy| policy.backend != backend) {
            return Err(crate::Error::Unsupported(format!(
                "decode-quality policy backend {:?} cannot authorize loaded backend {:?}",
                foreign.backend, backend
            )));
        }
        let selected = policies
            .into_iter()
            .filter(|policy| policy.tier == tier && policy.load_shape == self.load_shape)
            .collect::<Vec<_>>();
        self.validate_decode_quality_runtime_identity(&selected)?;
        Ok(selected)
    }

    /// Read-only view of the prepared File identities carried by this spec.
    ///
    /// The whole field is private so a finalized spec cannot be silently downgraded by replacing
    /// its collection with [`PreparedFilePins::default`].
    ///
    /// ```
    /// use gen_core::PreparedFilePins;
    /// assert!(!PreparedFilePins::default().is_prepared());
    /// ```
    ///
    /// ```compile_fail
    /// use gen_core::{LoadSpec, PreparedFilePins, WeightsSource};
    /// let mut spec = LoadSpec::new(WeightsSource::Dir("snapshot".into()));
    /// spec.prepared_file_pins = PreparedFilePins::default();
    /// ```
    pub fn prepared_file_pins(&self) -> &PreparedFilePins {
        &self.prepared_file_pins
    }

    /// Atomically install an already-pinned, complete File identity set without re-pinning.
    ///
    /// The caller can pin each source first, root-confine both its lexical
    /// [`PinnedWeightsFile::loader_path`] and exact [`PinnedWeightsFile::canonical_target_path`],
    /// then pass those same tokens here. The candidate set is fully checked before this spec is
    /// mutated. Tokens must name either configured File slots or descendants of configured Dir
    /// slots. An empty iterator intentionally finalizes a Dir-only spec.
    pub fn prepare_with_file_pins(
        &mut self,
        prepared: impl IntoIterator<Item = PinnedWeightsFile>,
    ) -> crate::Result<()> {
        let mut pins = BTreeMap::new();
        for pin in prepared {
            pin.verify_unchanged()?;
            let path = pin.loader_path().to_path_buf();
            if pins.insert(path.clone(), pin).is_some() {
                return Err(crate::Error::Unsupported(format!(
                    "duplicate prepared file token for {}",
                    path.display()
                )));
            }
        }
        let candidate = PreparedFilePins {
            prepared: true,
            finalized: true,
            pins: pins.into_iter().collect(),
        };
        if !self.prepared_receipt_paths.is_empty() || self.prepared_encoder_receipt.is_some() {
            return Err(crate::Error::Unsupported(
                "validated receipt identity must be installed through its owning contract".into(),
            ));
        }
        self.validate_prepared_file_pin_set_for(&candidate)?;
        if self.prepared_file_pins.is_finalized() && self.prepared_file_pins != candidate {
            return Err(crate::Error::Unsupported(
                "cannot replace finalized LoadSpec File identity".into(),
            ));
        }
        self.prepared_file_pins = candidate;
        Ok(())
    }

    /// Install the complete exact-file set retained by a crate-validated component receipt.
    ///
    /// Kept crate-private so an external caller cannot bless arbitrary paths as cache/load identity.
    pub(crate) fn prepare_with_validated_receipt_pins(
        &mut self,
        prepared: impl IntoIterator<Item = PinnedWeightsFile>,
        receipt_paths: impl IntoIterator<Item = PathBuf>,
        encoder_receipt: crate::encoder_contract::PreparedEncoderLoadReceipt,
    ) -> crate::Result<()> {
        let mut pins = BTreeMap::new();
        for pin in prepared {
            pin.verify_unchanged()?;
            let path = pin.loader_path().to_path_buf();
            if let Some(existing) = pins.insert(path.clone(), pin) {
                if existing != pins[&path] {
                    return Err(crate::Error::Unsupported(format!(
                        "conflicting prepared file tokens for {}",
                        path.display()
                    )));
                }
            }
        }
        let receipt_paths = receipt_paths.into_iter().collect::<BTreeSet<_>>();
        let candidate = PreparedFilePins {
            prepared: true,
            finalized: true,
            pins: pins.into_iter().collect(),
        };
        let previous_receipt_paths =
            std::mem::replace(&mut self.prepared_receipt_paths, receipt_paths);
        let previous_encoder_receipt = self.prepared_encoder_receipt.replace(encoder_receipt);
        if let Err(error) = self.validate_prepared_file_pin_set_for(&candidate) {
            self.prepared_receipt_paths = previous_receipt_paths;
            self.prepared_encoder_receipt = previous_encoder_receipt;
            return Err(error);
        }
        if self.prepared_file_pins.is_finalized()
            && (self.prepared_file_pins != candidate
                || self.prepared_receipt_paths != previous_receipt_paths
                || self.prepared_encoder_receipt != previous_encoder_receipt)
        {
            self.prepared_receipt_paths = previous_receipt_paths;
            self.prepared_encoder_receipt = previous_encoder_receipt;
            return Err(crate::Error::Unsupported(
                "cannot replace finalized LoadSpec receipt identity".into(),
            ));
        }
        self.prepared_file_pins = candidate;
        Ok(())
    }

    pub(crate) fn visit_source_paths<'a>(&'a self, mut visit: impl FnMut(&'a Path, bool)) {
        fn source<'a>(value: &'a WeightsSource, visit: &mut impl FnMut(&'a Path, bool)) {
            match value {
                WeightsSource::File(path) => visit(path, false),
                WeightsSource::Dir(path) => visit(path, true),
            }
        }
        source(&self.weights, &mut visit);
        if let Some(value) = &self.control {
            source(value, &mut visit);
        }
        for value in &self.extra_controls {
            source(value, &mut visit);
        }
        if let Some(value) = &self.ip_adapter {
            source(value, &mut visit);
        }
        for adapter in &self.adapters {
            visit(&adapter.path, false);
        }
        if let Some(pid) = &self.pid {
            source(&pid.checkpoint, &mut visit);
            source(&pid.gemma, &mut visit);
        }
        if let Some(identity) = &self.identity {
            for value in [&identity.encoder, &identity.eva, &identity.face_dir]
                .into_iter()
                .flatten()
            {
                source(value, &mut visit);
            }
        }
        if let Some(value) = &self.text_encoder {
            source(value, &mut visit);
        }
        for value in self.components.values() {
            source(value, &mut visit);
        }
    }

    /// Every File path configured anywhere on this load spec, in deterministic field order.
    ///
    /// This includes primary weights, typed overlays, adapters, PiD/identity/text-encoder files, and
    /// named components. Duplicate lexical paths occur once in [`Self::prepared_file_pins`].
    pub fn file_source_paths(&self) -> Vec<&Path> {
        fn push_source<'a>(paths: &mut Vec<&'a Path>, source: &'a WeightsSource) {
            if let WeightsSource::File(path) = source {
                paths.push(path.as_path());
            }
        }

        let mut paths = Vec::new();
        push_source(&mut paths, &self.weights);
        if let Some(source) = &self.control {
            push_source(&mut paths, source);
        }
        for source in &self.extra_controls {
            push_source(&mut paths, source);
        }
        if let Some(source) = &self.ip_adapter {
            push_source(&mut paths, source);
        }
        paths.extend(self.adapters.iter().map(|adapter| adapter.path.as_path()));
        if let Some(pid) = &self.pid {
            push_source(&mut paths, &pid.checkpoint);
            push_source(&mut paths, &pid.gemma);
        }
        if let Some(identity) = &self.identity {
            for source in [&identity.encoder, &identity.eva, &identity.face_dir]
                .into_iter()
                .flatten()
            {
                push_source(&mut paths, source);
            }
        }
        if let Some(source) = &self.text_encoder {
            push_source(&mut paths, source);
        }
        for source in self.components.values() {
            push_source(&mut paths, source);
        }
        paths
    }

    /// Every configured directory source, used to validate prepared member-file tokens.
    pub fn directory_source_paths(&self) -> Vec<&Path> {
        fn push_source<'a>(paths: &mut Vec<&'a Path>, source: &'a WeightsSource) {
            if let WeightsSource::Dir(path) = source {
                paths.push(path.as_path());
            }
        }

        let mut paths = Vec::new();
        push_source(&mut paths, &self.weights);
        if let Some(source) = &self.control {
            push_source(&mut paths, source);
        }
        for source in &self.extra_controls {
            push_source(&mut paths, source);
        }
        if let Some(source) = &self.ip_adapter {
            push_source(&mut paths, source);
        }
        if let Some(pid) = &self.pid {
            push_source(&mut paths, &pid.checkpoint);
            push_source(&mut paths, &pid.gemma);
        }
        if let Some(identity) = &self.identity {
            for source in [&identity.encoder, &identity.eva, &identity.face_dir]
                .into_iter()
                .flatten()
            {
                push_source(&mut paths, source);
            }
        }
        if let Some(source) = &self.text_encoder {
            push_source(&mut paths, source);
        }
        for source in self.components.values() {
            push_source(&mut paths, source);
        }
        paths
    }

    /// Attach one caller-prepared token to its expected configured File path.
    pub fn with_prepared_file_pin(
        mut self,
        expected_path: impl AsRef<Path>,
        prepared: PinnedWeightsFile,
    ) -> crate::Result<Self> {
        self.set_prepared_file_pin(expected_path, prepared)?;
        Ok(self)
    }

    /// Mutable counterpart to [`Self::with_prepared_file_pin`].
    pub fn set_prepared_file_pin(
        &mut self,
        expected_path: impl AsRef<Path>,
        prepared: PinnedWeightsFile,
    ) -> crate::Result<()> {
        let expected = std::path::absolute(expected_path.as_ref())?;
        let configured = self
            .file_source_paths()
            .into_iter()
            .map(std::path::absolute)
            .collect::<std::io::Result<Vec<_>>>()?;
        if !configured.contains(&expected) {
            return Err(crate::Error::Unsupported(format!(
                "prepared file token path is not configured on this LoadSpec: {}",
                expected.display()
            )));
        }
        if prepared.loader_path() != expected {
            return Err(crate::Error::Unsupported(format!(
                "prepared file token path mismatch: LoadSpec names {}, token names {}",
                expected.display(),
                prepared.loader_path().display()
            )));
        }
        prepared.verify_unchanged()?;
        if let Some(existing) = self.prepared_file_pins.get(&expected) {
            if existing != &prepared {
                return Err(crate::Error::Unsupported(format!(
                    "prepared file token was replaced for {}",
                    expected.display()
                )));
            }
            return Ok(());
        }
        if self.prepared_file_pins.finalized {
            return Err(crate::Error::Unsupported(format!(
                "cannot add prepared file token after LoadSpec File identity was finalized: {}",
                expected.display()
            )));
        }
        self.prepared_file_pins.prepared = true;
        let index = self
            .prepared_file_pins
            .pins
            .binary_search_by(|(path, _)| path.cmp(&expected))
            .unwrap_or_else(|index| index);
        self.prepared_file_pins
            .pins
            .insert(index, (expected, prepared));
        Ok(())
    }

    /// Pin every configured File source once, retaining any already attached caller token.
    ///
    /// Consumers call this after root confinement and before cache-key construction. Repeated calls
    /// validate existing tokens rather than replacing them.
    pub fn prepare_file_sources(&mut self) -> crate::Result<()> {
        if self.prepared_file_pins.is_finalized() {
            return self.validate_prepared_file_pins();
        }
        // Transition before touching the filesystem. A mid-prepare error leaves a sticky, partial
        // prepared spec that fails closed instead of falling back to on-demand re-pinning.
        self.prepared_file_pins.prepared = true;
        let paths: Vec<PathBuf> = self
            .file_source_paths()
            .into_iter()
            .map(Path::to_path_buf)
            .collect();
        for path in paths {
            let absolute = std::path::absolute(&path)?;
            if let Some(prepared) = self.prepared_file_pins.get(&absolute) {
                if prepared.loader_path() != absolute {
                    return Err(crate::Error::Unsupported(format!(
                        "prepared file token path mismatch: LoadSpec names {}, token names {}",
                        absolute.display(),
                        prepared.loader_path().display()
                    )));
                }
                prepared.verify_unchanged()?;
            } else {
                let prepared = PinnedWeightsFile::pin(&path)?;
                self.set_prepared_file_pin(&path, prepared)?;
            }
        }
        self.finish_file_source_preparation()
    }

    /// Finalize a complete caller-installed token set without re-pinning any source.
    ///
    /// This is the manual counterpart to [`Self::prepare_file_sources`]: a consumer can pin one
    /// source, validate [`PinnedWeightsFile::canonical_target_path`] against its allowed root, install
    /// that exact token with [`Self::set_prepared_file_pin`], repeat for every File slot, and call
    /// this method once. It is also how a caller explicitly prepares a Dir-only spec.
    pub fn finish_file_source_preparation(&mut self) -> crate::Result<()> {
        if self.prepared_file_pins.is_finalized() {
            return self.validate_prepared_file_pins();
        }
        self.prepared_file_pins.prepared = true;
        self.validate_prepared_file_pin_set_for(&self.prepared_file_pins)?;
        self.prepared_file_pins.finalized = true;
        Ok(())
    }

    /// Validate that prepared mode covers exactly the File sources currently configured by the spec.
    pub fn validate_prepared_file_pins(&self) -> crate::Result<()> {
        if let Some(guard) = &self.source_inventory {
            guard.verify_for(self)?;
        }
        if !self.prepared_file_pins.is_prepared() {
            return Ok(());
        }
        if !self.prepared_file_pins.is_finalized() {
            return Err(crate::Error::Unsupported(
                "LoadSpec File identity preparation has not been finalized".into(),
            ));
        }
        self.validate_prepared_file_pin_set_for(&self.prepared_file_pins)
    }

    /// Return path-free planning facts from the retained selected-encoder acquisition receipt.
    ///
    /// Bind all configured directory members and File/auxiliary sources to one retained inventory.
    /// Providers must use `read_prepared_files_unchanged` around deferred reads/materialization.
    pub fn bind_source_inventory(
        &mut self,
        inventory: crate::SourceInventory,
    ) -> crate::Result<()> {
        inventory.verify_for(self)?;
        // Retain the original configured File tokens in the standard loader path as well; do not
        // re-pin a fused checkpoint/override when a later phase consumes it. Directory members
        // additionally need the inventory guard (a configured-File set cannot detect new shards).
        if !self.prepared_file_pins.is_prepared() {
            self.prepare_with_file_pins(inventory.configured_file_pins(self)?)?;
        }
        self.source_inventory = Some(std::sync::Arc::new(inventory));
        Ok(())
    }

    pub fn source_inventory(&self) -> Option<&crate::SourceInventory> {
        self.source_inventory.as_deref()
    }

    /// Stable receipts are checked through their metadata identity only; this never acquires a new
    /// artifact seal or repeats full-content hashing. A changed source still fails closed before
    /// facts are returned. `None` means this spec has no crate-validated encoder receipt and a
    /// compatibility planner must use bounded discovery instead.
    pub fn prepared_text_encoder_planning_facts(
        &self,
    ) -> crate::Result<Option<crate::encoder_contract::TextEncoderPlanningFacts>> {
        self.prepared_encoder_receipt
            .as_ref()
            .map(|receipt| receipt.planning_facts_for(self.text_encoder.as_ref()))
            .transpose()
    }

    fn validate_prepared_file_pin_set_for(&self, prepared: &PreparedFilePins) -> crate::Result<()> {
        if let Some(receipt) = &self.prepared_encoder_receipt {
            receipt.ensure_unchanged_for(self.text_encoder.as_ref())?;
        }
        let mut expected: Vec<PathBuf> = self
            .file_source_paths()
            .into_iter()
            .map(std::path::absolute)
            .collect::<std::io::Result<_>>()?;
        expected.extend(self.prepared_receipt_paths.iter().cloned());
        expected.sort();
        expected.dedup();
        let configured_directories = self
            .directory_source_paths()
            .into_iter()
            .map(std::path::absolute)
            .collect::<std::io::Result<Vec<_>>>()?;
        let actual: Vec<PathBuf> = prepared.keys().cloned().collect();
        let missing = expected
            .iter()
            .filter(|path| !actual.contains(path))
            .cloned()
            .collect::<Vec<_>>();
        let orphaned = actual
            .iter()
            .filter(|path| {
                !expected.contains(path)
                    && !configured_directories
                        .iter()
                        .any(|directory| *path != directory && path.starts_with(directory))
            })
            .cloned()
            .collect::<Vec<_>>();
        if !missing.is_empty() || !orphaned.is_empty() {
            return Err(crate::Error::Unsupported(format!(
                "prepared file-token set does not match the LoadSpec sources: missing {missing:?}, orphaned {orphaned:?}"
            )));
        }
        for path in actual {
            let prepared = prepared.get(&path).ok_or_else(|| {
                crate::Error::Unsupported(format!(
                    "prepared file-token set is missing configured source {}",
                    path.display()
                ))
            })?;
            if prepared.loader_path() != path {
                return Err(crate::Error::Unsupported(format!(
                    "prepared file token path mismatch: LoadSpec names {}, token names {}",
                    path.display(),
                    prepared.loader_path().display()
                )));
            }
            prepared.verify_unchanged()?;
        }
        Ok(())
    }

    /// Return the exact caller-prepared token for `path`, if this is a prepared spec.
    ///
    /// Once the spec enters prepared mode, a missing token for a configured File fails closed. The
    /// expected path and token loader path are compared as absolute-but-lexical paths on every use.
    pub fn prepared_file_pin_for(
        &self,
        path: impl AsRef<Path>,
    ) -> crate::Result<Option<&PinnedWeightsFile>> {
        let expected = std::path::absolute(path.as_ref())?;
        let configured = self
            .file_source_paths()
            .into_iter()
            .map(std::path::absolute)
            .collect::<std::io::Result<Vec<_>>>()?;
        if !configured.contains(&expected) {
            return Err(crate::Error::Unsupported(format!(
                "requested file pin is not configured on this LoadSpec: {}",
                expected.display()
            )));
        }
        if !self.prepared_file_pins.is_prepared() {
            return Ok(None);
        }
        if !self.prepared_file_pins.is_finalized() {
            return Err(crate::Error::Unsupported(
                "LoadSpec File identity preparation has not been finalized".into(),
            ));
        }
        let prepared = self.prepared_file_pins.get(&expected).ok_or_else(|| {
            crate::Error::Unsupported(format!(
                "prepared file-token set is missing configured source {}",
                expected.display()
            ))
        })?;
        if prepared.loader_path() != expected {
            return Err(crate::Error::Unsupported(format!(
                "prepared file token path mismatch: LoadSpec names {}, token names {}",
                expected.display(),
                prepared.loader_path().display()
            )));
        }
        prepared.verify_unchanged()?;
        Ok(Some(prepared))
    }

    /// Resolve the exact file token a provider must retain or guard while it opens `path`.
    ///
    /// Prepared mode clones the caller token. An ordinary, unprepared spec pins the current source
    /// on demand for backward compatibility.
    pub fn file_pin_for(&self, path: impl AsRef<Path>) -> crate::Result<PinnedWeightsFile> {
        match self.prepared_file_pin_for(path.as_ref())? {
            Some(prepared) => Ok(prepared.clone()),
            None => PinnedWeightsFile::pin(path),
        }
    }

    /// Read `path` through its exact prepared token when this spec has been prepared.
    ///
    /// Ordinary compatibility-mode specs invoke `read` directly. This preserves providers' legacy
    /// validation and error ordering while making every prepared caller consume the token installed
    /// by the resolver instead of re-pinning the current pathname.
    pub fn read_file_unchanged_if_prepared<T, E>(
        &self,
        path: impl AsRef<Path>,
        read: impl FnOnce(&Path) -> std::result::Result<T, E>,
    ) -> std::result::Result<T, E>
    where
        E: From<crate::Error>,
    {
        match self.prepared_file_pin_for(path.as_ref()).map_err(E::from)? {
            Some(prepared) => __observe_input_success!(prepared.read_unchanged(read), |_| {
                record_sealed_input_read(
                    "gen-core::LoadSpec::read_file_unchanged_if_prepared",
                    std::iter::once(prepared),
                );
            }),
            None => read(path.as_ref()),
        }
    }

    /// Resolve the primary single-file token; directory-backed specs return `Ok(None)`.
    pub fn weights_file_pin(&self) -> crate::Result<Option<PinnedWeightsFile>> {
        match &self.weights {
            WeightsSource::Dir(_) => Ok(None),
            WeightsSource::File(path) => self.file_pin_for(path).map(Some),
        }
    }

    /// Execute a provider read while every listed File source is guarded by this spec's exact pins.
    ///
    /// All pins are checked before and after `read`; a post-read mutation error wins over a provider
    /// error, matching [`PinnedWeightsFile::read_unchanged`]. This lets batch adapter loaders retain
    /// their all-files ordering and zero-match semantics while consuming the same tokens as cache
    /// identity.
    pub fn read_files_unchanged<T, E, P>(
        &self,
        paths: impl IntoIterator<Item = P>,
        read: impl FnOnce() -> std::result::Result<T, E>,
    ) -> std::result::Result<T, E>
    where
        P: AsRef<Path>,
        E: From<crate::Error>,
    {
        let pins = paths
            .into_iter()
            .map(|path| self.file_pin_for(path).map_err(E::from))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        for pin in &pins {
            pin.verify_unchanged().map_err(E::from)?;
        }
        let result = read();
        for pin in &pins {
            pin.verify_unchanged().map_err(E::from)?;
        }
        #[cfg(feature = "model-inference-proof")]
        if result.is_ok() {
            record_sealed_input_read("gen-core::LoadSpec::read_files_unchanged", pins.iter());
        }
        result
    }

    /// Execute a registry/provider callback under the complete prepared File identity set.
    ///
    /// Ordinary compatibility-mode specs invoke `read` directly, preserving historical callback
    /// behavior and error ordering without introducing eager filesystem access. Prepared specs must
    /// cover every current File slot, may additionally cover configured directory members, and every
    /// token is checked before and after the callback so persistent and identity-changing replacement
    /// during the callback fails closed.
    /// As with [`PinnedWeightsFile::read_unchanged`], a fully restored original pathname object is
    /// outside this path-token guarantee.
    /// Explicit guarded counterpart for metadata readers. Do not enter the legacy directory
    /// verification/canonicalization path after a one-time retained-count check.
    pub fn read_prepared_files_unchanged_with_capacity<T, E: From<crate::Error>>(
        &self,
        next: &mut dyn FnMut(u64) -> crate::Result<()>,
        read: impl FnOnce(&mut dyn FnMut(u64) -> crate::Result<()>) -> std::result::Result<T, E>,
    ) -> std::result::Result<T, E> {
        let Some(inventory) = self.source_inventory() else {
            return crate::SourceInventory::metadata_gap(
                next,
                crate::SourceInventory::MISSING_INVENTORY_METADATA_GAP,
            )
            .map_err(E::from);
        };
        inventory
            .verify_for_with_capacity(self, next)
            .map_err(E::from)?;
        let result = inventory.read_unchanged_with_capacity(next, read);
        #[cfg(target_vendor = "apple")]
        self.validate_prepared_file_pins_with_capacity(next)
            .map_err(E::from)?;
        result
    }

    /// Validate the existing prepared set and encoder/tokenizer receipt using only
    /// bounded metadata operations. Every original missing/orphan/identity predicate
    /// is retained; no current-directory cardinality authorizes an allocation.
    #[cfg(target_vendor = "apple")]
    pub(crate) fn validate_prepared_file_pins_with_capacity(
        &self,
        next: &mut dyn FnMut(u64) -> crate::Result<()>,
    ) -> crate::Result<()> {
        use crate::source_inventory::bounded_fs as fs;
        if let Some(receipt) = &self.prepared_encoder_receipt {
            receipt.ensure_unchanged_for_with_capacity(self.text_encoder.as_ref(), next)?;
        }
        if !self.prepared_file_pins.is_prepared() {
            return Ok(());
        }
        if !self.prepared_file_pins.is_finalized() {
            return crate::SourceInventory::metadata_gap(
                next,
                "LoadSpec File identity preparation has not been finalized",
            );
        }
        let mut configured = Vec::new();
        let mut result: crate::Result<()> = Ok(());
        self.visit_source_paths(|path, directory| {
            if result.is_ok() {
                result = (|| {
                    let path = fs::absolute(path, next)?;
                    fs::push(&mut configured, (path, directory), next)
                })();
            }
        });
        result?;
        for (path, directory) in &configured {
            if !directory && self.prepared_file_pins.get(path).is_none() {
                return crate::SourceInventory::metadata_gap(
                    next,
                    "prepared file-token set is missing configured source",
                );
            }
        }
        for path in &self.prepared_receipt_paths {
            if self.prepared_file_pins.get(path).is_none() {
                return crate::SourceInventory::metadata_gap(
                    next,
                    "prepared file-token set is missing receipt source",
                );
            }
        }
        for (path, pin) in self.prepared_file_pins.iter() {
            let configured = configured.iter().any(|(root, directory)| {
                if *directory {
                    path != root && path.starts_with(root)
                } else {
                    path == root
                }
            });
            if pin.loader_path() != path
                || (!configured && !self.prepared_receipt_paths.contains(path))
            {
                return crate::SourceInventory::metadata_gap(
                    next,
                    "prepared file-token set does not match the LoadSpec sources",
                );
            }
            pin.verify_unchanged_with_capacity(next)?;
        }
        Ok(())
    }

    /// Bind a source closure with exact Vec ownership and actual ArcInner payload
    /// layouts checked BEFORE allocating. Native APIs keep their existing behavior.
    pub fn bind_source_inventory_with_capacity(
        &mut self,
        inventory: crate::SourceInventory,
        next: &mut dyn FnMut(u64) -> crate::Result<()>,
    ) -> crate::Result<()> {
        #[cfg(target_vendor = "apple")]
        {
            inventory.verify_for_with_capacity(self, next)?;
            self.validate_prepared_file_pins_with_capacity(next)?;
            let prepared = if self.prepared_file_pins.is_prepared() {
                None
            } else {
                Some(PreparedFilePins {
                    prepared: true,
                    finalized: true,
                    pins: inventory.configured_file_pins_with_capacity(self, next)?,
                })
            };
            next(bounded_arc_layout::<crate::SourceInventory>()?)?;
            let inventory = Arc::new(inventory);
            // Only commit after every fallible check/allocation preflight succeeds.
            if let Some(prepared) = prepared {
                self.prepared_file_pins = prepared;
            }
            self.source_inventory = Some(inventory);
            Ok(())
        }
        #[cfg(not(target_vendor = "apple"))]
        {
            let _ = inventory;
            crate::SourceInventory::metadata_gap(
                next,
                crate::SourceInventory::PIN_CAPTURE_METADATA_GAP,
            )
        }
    }

    pub fn read_prepared_files_unchanged<T, E>(
        &self,
        read: impl FnOnce() -> std::result::Result<T, E>,
    ) -> std::result::Result<T, E>
    where
        E: From<crate::Error>,
    {
        if !self.prepared_file_pins.is_prepared() {
            return match &self.source_inventory {
                Some(guard) => {
                    guard.verify_for(self).map_err(E::from)?;
                    guard.read_unchanged(read)
                }
                None => read(),
            };
        }
        self.validate_prepared_file_pins().map_err(E::from)?;
        let pins = self
            .prepared_file_pins
            .iter()
            .map(|(_, pin)| pin)
            .collect::<Vec<_>>();
        for pin in &pins {
            pin.verify_unchanged().map_err(E::from)?;
        }
        let result = read();
        for pin in &pins {
            pin.verify_unchanged().map_err(E::from)?;
        }
        if let Some(guard) = &self.source_inventory {
            guard.verify_unchanged().map_err(E::from)?;
        }
        #[cfg(feature = "model-inference-proof")]
        if result.is_ok() {
            record_sealed_input_read(
                "gen-core::LoadSpec::read_prepared_files_unchanged",
                pins.iter().copied(),
            );
        }
        result
    }

    /// Builder-style quantization override.
    pub fn with_quant(mut self, quant: Quant) -> Self {
        self.quantize = Some(quant);
        self
    }

    /// Select YuE2's AR execution mode independently of its weight tier.
    pub fn with_yue2_ar_mode(mut self, mode: Yue2ArMode) -> Self {
        self.yue2_ar_mode = mode;
        self
    }

    /// Select YuE2's stage compute policy independently of the weight tier.
    pub fn with_yue2_compute_policy(mut self, policy: Yue2ComputePolicy) -> Self {
        self.yue2_compute_policy = policy;
        self
    }

    /// Builder-style external text-encoder substitution.
    pub fn with_text_encoder(mut self, text_encoder: WeightsSource) -> Self {
        self.text_encoder = Some(text_encoder);
        self
    }

    /// Builder-style component-residency override (epic 10765, sc-10821). [`OffloadPolicy::Sequential`]
    /// asks a supporting provider to load→use→drop each heavy component to cap peak VRAM; the default
    /// [`OffloadPolicy::Resident`] keeps everything co-resident. Which engines honor it is advertised by
    /// [`Capabilities::supports_sequential_offload`](crate::generator::Capabilities::supports_sequential_offload).
    pub fn with_offload_policy(mut self, offload_policy: OffloadPolicy) -> Self {
        self.offload_policy = offload_policy;
        self
    }

    /// Builder-style materialization-shape override (SC-15998).
    ///
    /// This does not alter [`Self::offload_policy`]: all four combinations are representable, so a
    /// resident generator may defer transformer blocks and a staged generator may use an eager
    /// per-phase load.
    pub fn with_load_shape(mut self, load_shape: LoadShape) -> Self {
        self.load_shape = load_shape;
        self
    }

    /// Atomically record declaration eligibility while retaining eager materialization.
    ///
    /// This state is used when the provider's deferred surface additionally requires a later exact
    /// residency choice. It is not transition authority: a generic shaper must ignore it. The
    /// caller may use
    /// [`with_applied_load_shape_declaration`](Self::with_applied_load_shape_declaration) only after
    /// re-evaluating the exact declaration and provider with that requirement satisfied.
    pub fn with_eligible_load_shape_declaration(mut self) -> Self {
        self.load_shape = LoadShape::EagerMaterialization;
        self.load_shape_declaration_result = LoadShapeDeclarationResult::Eligible;
        self
    }

    /// Atomically record that an exact declaration evaluation admitted deferred materialization.
    ///
    /// [`LoadShapeDeclarationResult::Eligible`] alone never authorizes this transition; callers
    /// must re-evaluate the exact declaration/provider coordinates after any later policy choice.
    pub fn with_applied_load_shape_declaration(mut self) -> Self {
        self.load_shape = LoadShape::DeferredMaterialization;
        self.load_shape_declaration_result = LoadShapeDeclarationResult::Applied;
        self
    }

    /// Atomically record that a declaration refused deferred materialization.
    ///
    /// The eager compatibility shape is retained, but the authoritative refusal prevents a later
    /// legacy shaper from re-admitting the route.
    pub fn with_refused_load_shape_declaration(mut self) -> Self {
        self.load_shape = LoadShape::EagerMaterialization;
        self.load_shape_declaration_result = LoadShapeDeclarationResult::Refused;
        self
    }

    /// Validate the declaration authority and materialization shape as one fail-closed contract.
    ///
    /// [`LoadShapeDeclarationResult::NotEvaluated`] permits either historical shape because legacy
    /// shapers may already have run. Declaration-aware callers must use the atomic builders above
    /// and may validate again after cache or cold-load reconstruction.
    pub fn validate_load_shape_declaration(&self) -> crate::Result<()> {
        match (self.load_shape_declaration_result, self.load_shape) {
            (LoadShapeDeclarationResult::NotEvaluated, _)
            | (LoadShapeDeclarationResult::Eligible, LoadShape::EagerMaterialization)
            | (LoadShapeDeclarationResult::Applied, LoadShape::DeferredMaterialization)
            | (LoadShapeDeclarationResult::Refused, LoadShape::EagerMaterialization) => Ok(()),
            (LoadShapeDeclarationResult::Eligible, LoadShape::DeferredMaterialization) => {
                Err("eligible load-shape declaration must retain eager materialization until its exact policy requirement is satisfied".into())
            }
            (LoadShapeDeclarationResult::Applied, LoadShape::EagerMaterialization) => {
                Err("applied load-shape declaration must use deferred materialization".into())
            }
            (LoadShapeDeclarationResult::Refused, LoadShape::DeferredMaterialization) => {
                Err("refused load-shape declaration must retain eager materialization".into())
            }
        }
    }

    /// Builder-style control-branch overlay (the ControlNet checkpoint over the base `weights`).
    pub fn with_control(mut self, control: WeightsSource) -> Self {
        self.control = Some(control);
        self
    }

    /// Builder-style named component (epic 13657) — stage the caller-provisioned local path for the
    /// component `id` into [`components`](Self::components). Mirrors [`with_control`](Self::with_control);
    /// the id is the stable key a provider reads at load via
    /// [`require_component`](crate::control::require_component). Re-inserting the same id replaces the
    /// prior path (last write wins). See [`components`](Self::components) for the id registry.
    pub fn with_component(mut self, id: impl Into<String>, src: WeightsSource) -> Self {
        self.components.insert(id.into(), src);
        self
    }

    /// Builder-style **additional** ControlNet checkpoint for MultiControlNet (sc-3378) — appends to
    /// [`extra_controls`](Self::extra_controls). Call after [`with_control`](Self::with_control); each
    /// extra branch pairs, in order, with the request's `Conditioning::Control` images. Supported by
    /// the SDXL provider.
    pub fn with_extra_control(mut self, control: WeightsSource) -> Self {
        self.extra_controls.push(control);
        self
    }

    /// Builder-style IP-Adapter overlay (the image-prompt checkpoint dir over the base `weights`).
    pub fn with_ip_adapter(mut self, ip_adapter: WeightsSource) -> Self {
        self.ip_adapter = Some(ip_adapter);
        self
    }

    /// Builder-style LoRA/LoKr adapters to bake on at load time (replaces any already set).
    pub fn with_adapters(mut self, adapters: Vec<AdapterSpec>) -> Self {
        self.adapters = adapters;
        self
    }

    /// Builder-style optional PiD decoder overlay (epic 7840) — the converted per-latent-space PiD
    /// checkpoint + the Gemma-2 caption-encoder snapshot dir. Makes PiD *available*; the per-request
    /// [`crate::GenerationRequest::use_pid`] flag then selects it at decode.
    pub fn with_pid(mut self, checkpoint: WeightsSource, gemma: WeightsSource) -> Self {
        self.pid = Some(PidWeights { checkpoint, gemma });
        self
    }
}
