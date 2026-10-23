//! SDXL source-context transport, separate from the existing full-content work digest.
//! The ledger bounds requested layouts only. Tokenizer/fixed-vocabulary work, malloc usable
//! sizes, process growth and physical realloc retirement are NOT certified by this mechanism.
use mlx_gen::gen_core::GenerationRequest;
use mlx_gen::{Error, Result};
use mlx_gen_sdxl::sampling::{
    SamplingSourceFacts, SamplingSourceFactsSeed, SamplingSourceView, SourceRequestBudget,
};
use serde::de::{DeserializeSeed, MapAccess, Visitor};
use serde::ser::{SerializeMap, SerializeStruct};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use sha2::{Digest, Sha256};
use std::alloc::Layout;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

const DOMAIN: &[u8] = b"ios-sdxl-source-context-v1\0";

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct SourceDigest([u8; 32]);
impl Serialize for SourceDigest {
    fn serialize<S: Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        let mut text = [0u8; 64];
        for (i, &b) in self.0.iter().enumerate() {
            text[2 * i] = b"0123456789abcdef"[(b >> 4) as usize];
            text[2 * i + 1] = b"0123456789abcdef"[(b & 15) as usize];
        }
        s.serialize_str(std::str::from_utf8(&text).expect("hex is ASCII"))
    }
}
impl<'de> Deserialize<'de> for SourceDigest {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        struct Hex;
        impl Visitor<'_> for Hex {
            type Value = SourceDigest;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("64 lowercase hex source-digest bytes")
            }
            fn visit_str<E: serde::de::Error>(
                self,
                s: &str,
            ) -> std::result::Result<Self::Value, E> {
                if s.len() != 64 {
                    return Err(E::custom("invalid source digest length"));
                }
                let digit = |b: u8| match b {
                    b'0'..=b'9' => Some(b - b'0'),
                    b'a'..=b'f' => Some(b - b'a' + 10),
                    _ => None,
                };
                let mut out = [0; 32];
                for (i, pair) in s.as_bytes().chunks_exact(2).enumerate() {
                    let hi = digit(pair[0])
                        .ok_or_else(|| E::custom("source digest is not lowercase hex"))?;
                    let lo = digit(pair[1])
                        .ok_or_else(|| E::custom("source digest is not lowercase hex"))?;
                    out[i] = (hi << 4) | lo;
                }
                Ok(SourceDigest(out))
            }
        }
        d.deserialize_str(Hex)
    }
}
impl fmt::Debug for SourceDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for b in self.0 {
            write!(f, "{b:02x}")?;
        }
        Ok(())
    }
}
fn digest(view: SamplingSourceView<'_>) -> SourceDigest {
    fn image(h: &mut Sha256, i: mlx_gen_sdxl::sampling::ReferenceImageFacts) {
        h.update(i.width.to_le_bytes());
        h.update(i.height.to_le_bytes());
        h.update(i.buffer_bytes.to_le_bytes());
    }
    fn index(h: &mut Sha256, i: Option<u64>) {
        h.update([u8::from(i.is_some())]);
        if let Some(i) = i {
            h.update(i.to_le_bytes());
        }
    }
    let mut h = Sha256::new();
    h.update(DOMAIN);
    let work = view.work();
    let text = view.text();
    for n in [work.reference_count, work.mask_count, work.control_count] {
        h.update((n as u64).to_le_bytes());
    }
    h.update([u8::from(work.reference_image.is_some())]);
    if let Some(i) = work.reference_image {
        image(&mut h, i);
    }
    h.update(text.prompt_utf8_bytes.to_le_bytes());
    h.update(text.negative_prompt_utf8_bytes.to_le_bytes());
    index(&mut h, view.reference_index());
    let mask = view.mask();
    h.update([u8::from(mask.is_some())]);
    if let Some(m) = mask {
        h.update(m.source_index.to_le_bytes());
        image(&mut h, m.image);
    }
    h.update((work.control_count as u64).to_le_bytes());
    for c in view.controls() {
        h.update(c.source_index.to_le_bytes());
        image(&mut h, c.image);
    }
    SourceDigest(h.finalize().into())
}

/// Private verified payload, shared by key/row clones. Ordinary deserialization can construct only
/// a digest-only query; a full imported payload MUST use BindingSeed with an explicit budget.
#[derive(Clone)]
pub struct SdxlWorkBinding {
    sha256: SourceDigest,
    facts: Option<Arc<SamplingSourceFacts>>,
}
impl SdxlWorkBinding {
    pub fn query(request: &GenerationRequest) -> Self {
        Self {
            sha256: digest(SamplingSourceView::from_request(request)),
            facts: None,
        }
    }
    pub fn capture(
        request: &GenerationRequest,
        budget: &mut SourceRequestBudget<'_>,
    ) -> Result<Self> {
        let facts =
            SamplingSourceFacts::capture(SamplingSourceView::from_request(request), budget)?;
        Self::verified(digest(facts.view()), facts, budget)
    }
    fn verified(
        sha256: SourceDigest,
        facts: SamplingSourceFacts,
        budget: &mut SourceRequestBudget<'_>,
    ) -> Result<Self> {
        facts.validate_roles()?;
        if digest(facts.view()) != sha256 {
            return Err(invalid("source digest and payload disagree"));
        }
        let request = arc_request_layout()?.size() as u64;
        budget.before_replace(0, request)?;
        let facts = Arc::new(facts); // fixed infallible OOM path; not a claim of fallible allocation
        budget.commit_replace(0, request);
        Ok(Self {
            sha256,
            facts: Some(facts),
        })
    }
    pub fn facts(&self) -> Option<&SamplingSourceFacts> {
        self.facts.as_deref()
    }
    pub fn validate_actual(&self, request: &GenerationRequest) -> Result<()> {
        if self.sha256 != digest(SamplingSourceView::from_request(request)) {
            return Err(invalid("source context differs from its actual request"));
        }
        Ok(())
    }
    pub fn require_facts(&self) -> Result<&SamplingSourceFacts> {
        self.facts().ok_or_else(|| {
            invalid("payload-free SDXL replay requires recorded image AND text source context")
        })
    }
}
impl fmt::Debug for SdxlWorkBinding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SdxlWorkBinding")
            .field("sha256", &self.sha256)
            .field("has_facts", &self.facts.is_some())
            .finish()
    }
}
impl PartialEq for SdxlWorkBinding {
    fn eq(&self, other: &Self) -> bool {
        self.sha256 == other.sha256
    }
}
impl Eq for SdxlWorkBinding {}
impl PartialOrd for SdxlWorkBinding {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for SdxlWorkBinding {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.sha256.cmp(&other.sha256)
    }
}
impl Hash for SdxlWorkBinding {
    fn hash<H: Hasher>(&self, h: &mut H) {
        self.sha256.hash(h)
    }
}
impl Serialize for SdxlWorkBinding {
    fn serialize<S: Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        let mut out =
            s.serialize_struct("SdxlWorkBinding", if self.facts.is_some() { 2 } else { 1 })?;
        out.serialize_field("sha256", &self.sha256)?;
        if let Some(facts) = self.facts.as_deref() {
            out.serialize_field("facts", facts)?;
        }
        out.end()
    }
}
#[derive(Deserialize)]
#[serde(field_identifier, rename_all = "snake_case")]
enum BindingField {
    Sha256,
    Facts,
}
impl<'de> Deserialize<'de> for SdxlWorkBinding {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        struct QueryOnly;
        impl<'de> Visitor<'de> for QueryOnly {
            type Value = SdxlWorkBinding;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("digest-only SDXL query; full facts require a budgeted seed")
            }
            fn visit_map<M: MapAccess<'de>>(
                self,
                mut map: M,
            ) -> std::result::Result<Self::Value, M::Error> {
                let mut sha256 = None;
                while let Some(field) = map.next_key::<BindingField>()? {
                    match field {
                        BindingField::Sha256 => {
                            if sha256.is_some() {
                                return Err(serde::de::Error::duplicate_field("sha256"));
                            }
                            sha256 = Some(map.next_value()?);
                        }
                        // Deliberately do not call next_value/deserialize_seq, even for facts:null.
                        BindingField::Facts => {
                            return Err(serde::de::Error::custom(
                                "full SDXL source facts require an explicit budgeted import",
                            ))
                        }
                    }
                }
                Ok(SdxlWorkBinding {
                    sha256: sha256.ok_or_else(|| serde::de::Error::missing_field("sha256"))?,
                    facts: None,
                })
            }
        }
        d.deserialize_map(QueryOnly)
    }
}
pub(crate) struct BindingSeed<'a, 'b> {
    pub budget: &'a mut SourceRequestBudget<'b>,
}
impl<'de> DeserializeSeed<'de> for BindingSeed<'_, '_> {
    type Value = SdxlWorkBinding;
    fn deserialize<D: Deserializer<'de>>(self, d: D) -> std::result::Result<Self::Value, D::Error> {
        d.deserialize_map(self)
    }
}
impl<'de> Visitor<'de> for BindingSeed<'_, '_> {
    type Value = SdxlWorkBinding;
    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("budgeted verified SDXL source binding")
    }
    fn visit_map<M: MapAccess<'de>>(
        self,
        mut map: M,
    ) -> std::result::Result<Self::Value, M::Error> {
        let (mut sha256, mut facts) = (None, None);
        while let Some(field) = map.next_key::<BindingField>()? {
            match field {
                BindingField::Sha256 => {
                    if sha256.is_some() {
                        return Err(serde::de::Error::duplicate_field("sha256"));
                    }
                    sha256 = Some(map.next_value()?);
                }
                BindingField::Facts => {
                    if facts.is_some() {
                        return Err(serde::de::Error::duplicate_field("facts"));
                    }
                    facts = Some(map.next_value_seed(SamplingSourceFactsSeed {
                        budget: &mut *self.budget,
                    })?);
                }
            }
        }
        let sha256 = sha256.ok_or_else(|| serde::de::Error::missing_field("sha256"))?;
        match facts {
            Some(f) => {
                SdxlWorkBinding::verified(sha256, f, self.budget).map_err(serde::de::Error::custom)
            }
            None => Ok(SdxlWorkBinding {
                sha256,
                facts: None,
            }),
        }
    }
}
pub(crate) struct OptionalBindingSeed<'a, 'b> {
    pub budget: &'a mut SourceRequestBudget<'b>,
}
impl<'de> DeserializeSeed<'de> for OptionalBindingSeed<'_, '_> {
    type Value = Option<SdxlWorkBinding>;
    fn deserialize<D: Deserializer<'de>>(self, d: D) -> std::result::Result<Self::Value, D::Error> {
        d.deserialize_option(self)
    }
}
impl<'de> Visitor<'de> for OptionalBindingSeed<'_, '_> {
    type Value = Option<SdxlWorkBinding>;
    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("optional budgeted source binding")
    }
    fn visit_none<E: serde::de::Error>(self) -> std::result::Result<Self::Value, E> {
        Ok(None)
    }
    fn visit_unit<E: serde::de::Error>(self) -> std::result::Result<Self::Value, E> {
        Ok(None)
    }
    fn visit_some<D: Deserializer<'de>>(self, d: D) -> std::result::Result<Self::Value, D::Error> {
        BindingSeed {
            budget: &mut *self.budget,
        }
        .deserialize(d)
        .map(Some)
    }
}
fn invalid(s: &str) -> Error {
    Error::Unsupported(format!("SDXL source transport: {s}"))
}

/// Corrected Rust ac68 source: ArcInner is repr(C,align(2)) with two atomic usize counters.
/// Derived from actual type sizes/alignments, not a runtime rustc-version or measurement gate.
/// On the documented 64-bit/Atomic8/8 cfg this header is 16/8. Requested layout ONLY.
fn arc_request_layout() -> Result<Layout> {
    let counters = Layout::new::<[AtomicUsize; 2]>();
    let alignment = counters.align().max(2);
    let header = Layout::from_size_align(counters.size(), alignment)
        .map_err(|_| invalid("Arc header layout overflow"))?;
    let (full, _) = header
        .extend(Layout::new::<SamplingSourceFacts>())
        .map_err(|_| invalid("Arc payload layout overflow"))?;
    Ok(full.pad_to_align())
}

/// Serialize the typed key at the final output boundary, not into an intermediate Value tree.
/// Count exact bytes, then reserve once and reject writes beyond that count. Existing retained
/// facts are in the ledger; the live hook receives only this NEXT String allocation request.
fn serialize_framed_with_budget<T: Serialize>(
    prefix: &str,
    suffix: &str,
    value: &T,
    budget: &mut SourceRequestBudget<'_>,
) -> Result<String> {
    use std::io::Write as _;
    struct Count(usize);
    impl std::io::Write for Count {
        fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
            self.0 = self
                .0
                .checked_add(b.len())
                .ok_or_else(|| std::io::Error::other("JSON size overflow"))?;
            Ok(b.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let framing = prefix
        .len()
        .checked_add(suffix.len())
        .ok_or_else(|| invalid("output framing overflow"))?;
    let mut count = Count(framing);
    serde_json::to_writer(&mut count, value).map_err(|e| invalid(&e.to_string()))?;
    let layout =
        Layout::array::<u8>(count.0).map_err(|_| invalid("JSON output layout overflow"))?;
    let next = layout.size() as u64;
    budget.before_replace(0, next)?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(count.0)
        .map_err(|_| invalid("JSON output reservation failed"))?;
    budget.commit_replace(0, next);
    struct Bounded<'a> {
        bytes: &'a mut Vec<u8>,
        limit: usize,
    }
    impl std::io::Write for Bounded<'_> {
        fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
            if b.len() > self.limit.saturating_sub(self.bytes.len()) {
                return Err(std::io::Error::other("JSON changed after counting"));
            }
            self.bytes.extend_from_slice(b);
            Ok(b.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut writer = Bounded {
        bytes: &mut bytes,
        limit: count.0,
    };
    writer
        .write_all(prefix.as_bytes())
        .map_err(|e| invalid(&e.to_string()))?;
    serde_json::to_writer(&mut writer, value).map_err(|e| invalid(&e.to_string()))?;
    writer
        .write_all(suffix.as_bytes())
        .map_err(|e| invalid(&e.to_string()))?;
    String::from_utf8(bytes).map_err(|_| invalid("JSON writer returned non-UTF8"))
}

/// Capture only from the key's own actual request, never compare a request against an anchor's
/// different geometry. Full-content identity remains the independent existing work_binding.
pub fn record_source_context(
    key: &mut crate::evidence::EvidenceKey,
    candidate: &crate::IosCandidate,
    budget: &mut SourceRequestBudget<'_>,
) -> Result<()> {
    if candidate.provider != crate::sdxl::SDXL_ID {
        return Ok(());
    }
    let crate::IosWork::Generate(request) = &candidate.work else {
        return Err(invalid("SDXL capture has no generation request"));
    };
    if let Some(binding) = &key.sdxl_work {
        binding.validate_actual(request)?;
    }
    let binding = SdxlWorkBinding::capture(request, budget)?;
    validate_key_roles(key, binding.require_facts()?)?;
    key.sdxl_work = Some(binding);
    Ok(())
}
pub(crate) fn validate_key_roles(
    key: &crate::evidence::EvidenceKey,
    facts: &SamplingSourceFacts,
) -> Result<()> {
    facts.validate_roles()?;
    let work = facts.work();
    if (work.reference_count, work.mask_count, work.control_count)
        != (
            crate::pricing::reference_role_count(
                key.reference_count,
                key.mask_count,
                key.control_count,
            )? as usize,
            key.mask_count as usize,
            key.control_count as usize,
        )
    {
        return Err(invalid(
            "recorded source facts and evidence key roles disagree",
        ));
    }
    Ok(())
}
impl SdxlWorkBinding {
    pub fn retained_requested_bytes(&self) -> Result<u64> {
        let Some(facts) = self.facts() else {
            return Ok(0);
        };
        let vector = Layout::array::<mlx_gen_sdxl::sampling::IndexedImageSourceFacts>(
            facts.controls().len(),
        )
        .map_err(|_| invalid("retained Control layout overflow"))?;
        (vector.size() as u64)
            .checked_add(arc_request_layout()?.size() as u64)
            .ok_or_else(|| invalid("retained source layout overflow"))
    }
}
struct CaptureOverlay<'a> {
    row: &'a serde_json::Map<String, serde_json::Value>,
    key: Option<&'a crate::evidence::EvidenceKey>,
}
impl Serialize for CaptureOverlay<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        let existing = self.row.contains_key("evidence_key");
        let mut map = s.serialize_map(Some(self.row.len() + usize::from(!existing)))?;
        let mut wrote = false;
        for (name, value) in self.row {
            if !wrote && name.as_str() >= "evidence_key" {
                map.serialize_entry("evidence_key", &self.key)?;
                wrote = true;
            }
            if name != "evidence_key" {
                map.serialize_entry(name, value)?;
            }
        }
        if !wrote {
            map.serialize_entry("evidence_key", &self.key)?;
        }
        map.end()
    }
}
/// Same capture JSON object/schema, with a borrowed typed key instead of Value materialization.
/// `prefix` is the existing SMOKE/CAMPAIGN_JSON framing, included in the exact output count.
pub fn serialize_capture_row(
    prefix: &str,
    row: &serde_json::Value,
    key: Option<&crate::evidence::EvidenceKey>,
    budget: &mut SourceRequestBudget<'_>,
) -> Result<String> {
    let row = row
        .as_object()
        .ok_or_else(|| invalid("capture row is not an object"))?;
    serialize_framed_with_budget(prefix, "\n", &CaptureOverlay { row, key }, budget)
}

struct KeySeed<'a, 'b> {
    budget: &'a mut SourceRequestBudget<'b>,
}
impl<'de> DeserializeSeed<'de> for KeySeed<'_, '_> {
    type Value = crate::evidence::EvidenceKey;
    fn deserialize<D: Deserializer<'de>>(self, d: D) -> std::result::Result<Self::Value, D::Error> {
        d.deserialize_map(self)
    }
}
impl<'de> Visitor<'de> for KeySeed<'_, '_> {
    type Value = crate::evidence::EvidenceKey;
    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("evidence key with explicit source budget")
    }
    fn visit_map<M: MapAccess<'de>>(
        self,
        mut map: M,
    ) -> std::result::Result<Self::Value, M::Error> {
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "snake_case")]
        enum Field {
            Machine,
            Provider,
            Route,
            Tier,
            LayoutSha256,
            ArchitectureSha256,
            ConfiguredVariantSha256,
            Mode,
            Width,
            Height,
            Batch,
            ReferenceCount,
            MaskCount,
            ControlCount,
            LoadForm,
            Execution,
            SamplingRecipe,
            SdxlWork,
            TokenWindow,
        }
        let mut machine = None;
        let mut provider = None;
        let mut route = None;
        let mut tier = None;
        let mut layout_sha256 = None;
        let mut architecture_sha256 = None;
        let mut configured_variant_sha256 = None;
        let mut mode = None;
        let mut width = None;
        let mut height = None;
        let mut batch = None;
        let mut reference_count = None;
        let mut mask_count = None;
        let mut control_count = None;
        let mut load_form = None;
        let mut execution = None;
        let mut sampling_recipe = None;
        let mut sdxl_work = None;
        let mut token_window = None;
        while let Some(field) = map.next_key::<Field>()? {
            match field {
                Field::Machine => {
                    if machine.is_some() {
                        return Err(serde::de::Error::duplicate_field("machine"));
                    }
                    machine = Some(map.next_value()?);
                }
                Field::Provider => {
                    if provider.is_some() {
                        return Err(serde::de::Error::duplicate_field("provider"));
                    }
                    provider = Some(map.next_value()?);
                }
                Field::Route => {
                    if route.is_some() {
                        return Err(serde::de::Error::duplicate_field("route"));
                    }
                    route = Some(map.next_value()?);
                }
                Field::Tier => {
                    if tier.is_some() {
                        return Err(serde::de::Error::duplicate_field("tier"));
                    }
                    tier = Some(map.next_value()?);
                }
                Field::LayoutSha256 => {
                    if layout_sha256.is_some() {
                        return Err(serde::de::Error::duplicate_field("layout_sha256"));
                    }
                    layout_sha256 = Some(map.next_value()?);
                }
                Field::ArchitectureSha256 => {
                    if architecture_sha256.is_some() {
                        return Err(serde::de::Error::duplicate_field("architecture_sha256"));
                    }
                    architecture_sha256 = Some(map.next_value()?);
                }
                Field::ConfiguredVariantSha256 => {
                    if configured_variant_sha256.is_some() {
                        return Err(serde::de::Error::duplicate_field(
                            "configured_variant_sha256",
                        ));
                    }
                    configured_variant_sha256 = Some(map.next_value()?);
                }
                Field::Mode => {
                    if mode.is_some() {
                        return Err(serde::de::Error::duplicate_field("mode"));
                    }
                    mode = Some(map.next_value()?);
                }
                Field::Width => {
                    if width.is_some() {
                        return Err(serde::de::Error::duplicate_field("width"));
                    }
                    width = Some(map.next_value()?);
                }
                Field::Height => {
                    if height.is_some() {
                        return Err(serde::de::Error::duplicate_field("height"));
                    }
                    height = Some(map.next_value()?);
                }
                Field::Batch => {
                    if batch.is_some() {
                        return Err(serde::de::Error::duplicate_field("batch"));
                    }
                    batch = Some(map.next_value()?);
                }
                Field::ReferenceCount => {
                    if reference_count.is_some() {
                        return Err(serde::de::Error::duplicate_field("reference_count"));
                    }
                    reference_count = Some(map.next_value()?);
                }
                Field::MaskCount => {
                    if mask_count.is_some() {
                        return Err(serde::de::Error::duplicate_field("mask_count"));
                    }
                    mask_count = Some(map.next_value()?);
                }
                Field::ControlCount => {
                    if control_count.is_some() {
                        return Err(serde::de::Error::duplicate_field("control_count"));
                    }
                    control_count = Some(map.next_value()?);
                }
                Field::LoadForm => {
                    if load_form.is_some() {
                        return Err(serde::de::Error::duplicate_field("load_form"));
                    }
                    load_form = Some(map.next_value()?);
                }
                Field::Execution => {
                    if execution.is_some() {
                        return Err(serde::de::Error::duplicate_field("execution"));
                    }
                    execution = Some(map.next_value()?);
                }
                Field::SamplingRecipe => {
                    if sampling_recipe.is_some() {
                        return Err(serde::de::Error::duplicate_field("sampling_recipe"));
                    }
                    sampling_recipe = Some(map.next_value()?);
                }
                Field::SdxlWork => {
                    if sdxl_work.is_some() {
                        return Err(serde::de::Error::duplicate_field("sdxl_work"));
                    }
                    sdxl_work = Some(map.next_value_seed(OptionalBindingSeed {
                        budget: &mut *self.budget,
                    })?);
                }
                Field::TokenWindow => {
                    if token_window.is_some() {
                        return Err(serde::de::Error::duplicate_field("token_window"));
                    }
                    token_window = Some(map.next_value()?);
                }
            }
        }
        Ok(crate::evidence::EvidenceKey {
            machine: machine.ok_or_else(|| serde::de::Error::missing_field("machine"))?,
            provider: provider.ok_or_else(|| serde::de::Error::missing_field("provider"))?,
            route: route.ok_or_else(|| serde::de::Error::missing_field("route"))?,
            tier: tier.ok_or_else(|| serde::de::Error::missing_field("tier"))?,
            layout_sha256: layout_sha256
                .ok_or_else(|| serde::de::Error::missing_field("layout_sha256"))?,
            architecture_sha256: architecture_sha256
                .ok_or_else(|| serde::de::Error::missing_field("architecture_sha256"))?,
            configured_variant_sha256: configured_variant_sha256.unwrap_or(None),
            mode: mode.ok_or_else(|| serde::de::Error::missing_field("mode"))?,
            width: width.ok_or_else(|| serde::de::Error::missing_field("width"))?,
            height: height.ok_or_else(|| serde::de::Error::missing_field("height"))?,
            batch: batch.ok_or_else(|| serde::de::Error::missing_field("batch"))?,
            reference_count: reference_count
                .ok_or_else(|| serde::de::Error::missing_field("reference_count"))?,
            mask_count: mask_count.ok_or_else(|| serde::de::Error::missing_field("mask_count"))?,
            control_count: control_count
                .ok_or_else(|| serde::de::Error::missing_field("control_count"))?,
            load_form: load_form.ok_or_else(|| serde::de::Error::missing_field("load_form"))?,
            execution: execution.ok_or_else(|| serde::de::Error::missing_field("execution"))?,
            sampling_recipe: sampling_recipe.unwrap_or(None),
            sdxl_work: sdxl_work.unwrap_or(None),
            token_window: token_window
                .ok_or_else(|| serde::de::Error::missing_field("token_window"))?,
        })
    }
}
struct RowSeed<'a, 'b> {
    budget: &'a mut SourceRequestBudget<'b>,
}
impl<'de> DeserializeSeed<'de> for RowSeed<'_, '_> {
    type Value = crate::evidence::AnchorRow;
    fn deserialize<D: Deserializer<'de>>(self, d: D) -> std::result::Result<Self::Value, D::Error> {
        d.deserialize_map(self)
    }
}
impl<'de> Visitor<'de> for RowSeed<'_, '_> {
    type Value = crate::evidence::AnchorRow;
    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("budgeted reviewed anchor row")
    }
    fn visit_map<M: MapAccess<'de>>(
        self,
        mut map: M,
    ) -> std::result::Result<Self::Value, M::Error> {
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "snake_case")]
        enum Field {
            Key,
            AnchorKind,
            Selection,
            Outcome,
            EvidenceClass,
        }
        let (mut key, mut anchor_kind, mut selection, mut outcome, mut evidence_class) =
            (None, None, None, None, None);
        while let Some(field) = map.next_key::<Field>()? {
            macro_rules! scalar {
                ($slot:ident,$name:literal) => {{
                    if $slot.is_some() {
                        return Err(serde::de::Error::duplicate_field($name));
                    }
                    $slot = Some(map.next_value()?);
                }};
            }
            match field {
                Field::Key => {
                    if key.is_some() {
                        return Err(serde::de::Error::duplicate_field("key"));
                    }
                    key = Some(map.next_value_seed(KeySeed {
                        budget: &mut *self.budget,
                    })?);
                }
                Field::AnchorKind => scalar!(anchor_kind, "anchor_kind"),
                Field::Selection => scalar!(selection, "selection"),
                Field::Outcome => scalar!(outcome, "outcome"),
                Field::EvidenceClass => scalar!(evidence_class, "evidence_class"),
            }
        }
        Ok(crate::evidence::AnchorRow {
            key: key.ok_or_else(|| serde::de::Error::missing_field("key"))?,
            anchor_kind: anchor_kind
                .ok_or_else(|| serde::de::Error::missing_field("anchor_kind"))?,
            selection: selection.ok_or_else(|| serde::de::Error::missing_field("selection"))?,
            outcome: outcome.ok_or_else(|| serde::de::Error::missing_field("outcome"))?,
            evidence_class: evidence_class
                .ok_or_else(|| serde::de::Error::missing_field("evidence_class"))?,
        })
    }
}
struct RowsSeed<'a, 'b> {
    budget: &'a mut SourceRequestBudget<'b>,
}
impl<'de> DeserializeSeed<'de> for RowsSeed<'_, '_> {
    type Value = Vec<crate::evidence::AnchorRow>;
    fn deserialize<D: Deserializer<'de>>(self, d: D) -> std::result::Result<Self::Value, D::Error> {
        d.deserialize_seq(self)
    }
}
impl<'de> Visitor<'de> for RowsSeed<'_, '_> {
    type Value = Vec<crate::evidence::AnchorRow>;
    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("reviewed anchor sequence")
    }
    fn visit_seq<S: serde::de::SeqAccess<'de>>(
        self,
        mut seq: S,
    ) -> std::result::Result<Self::Value, S::Error> {
        let mut rows = Vec::new();
        while let Some(row) = seq.next_element_seed(RowSeed {
            budget: &mut *self.budget,
        })? {
            let needed = rows
                .len()
                .checked_add(1)
                .ok_or_else(|| serde::de::Error::custom("row count overflow"))?;
            if needed > rows.capacity() {
                let old = Layout::array::<crate::evidence::AnchorRow>(rows.capacity())
                    .map_err(serde::de::Error::custom)?
                    .size() as u64;
                let next = Layout::array::<crate::evidence::AnchorRow>(needed)
                    .map_err(serde::de::Error::custom)?
                    .size() as u64;
                self.budget
                    .before_replace(old, next)
                    .map_err(serde::de::Error::custom)?;
                rows.try_reserve_exact(needed - rows.len())
                    .map_err(serde::de::Error::custom)?;
                self.budget.commit_replace(old, next);
            }
            rows.push(row);
        }
        Ok(rows)
    }
}
struct TableSeed<'a, 'b> {
    budget: &'a mut SourceRequestBudget<'b>,
}
impl<'de> DeserializeSeed<'de> for TableSeed<'_, '_> {
    type Value = (u32, Vec<crate::evidence::AnchorRow>);
    fn deserialize<D: Deserializer<'de>>(self, d: D) -> std::result::Result<Self::Value, D::Error> {
        d.deserialize_map(self)
    }
}
impl<'de> Visitor<'de> for TableSeed<'_, '_> {
    type Value = (u32, Vec<crate::evidence::AnchorRow>);
    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("schema3 reviewed evidence table")
    }
    fn visit_map<M: MapAccess<'de>>(
        self,
        mut map: M,
    ) -> std::result::Result<Self::Value, M::Error> {
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "snake_case")]
        enum Field {
            Schema,
            About,
            Anchors,
        }
        let (mut schema, mut about, mut anchors) = (None, false, None);
        while let Some(field) = map.next_key::<Field>()? {
            match field {
                Field::Schema => {
                    if schema.is_some() {
                        return Err(serde::de::Error::duplicate_field("schema"));
                    }
                    schema = Some(map.next_value()?);
                }
                Field::About => {
                    if about {
                        return Err(serde::de::Error::duplicate_field("about"));
                    }
                    let _: String = map.next_value()?;
                    about = true;
                }
                Field::Anchors => {
                    if anchors.is_some() {
                        return Err(serde::de::Error::duplicate_field("anchors"));
                    }
                    anchors = Some(map.next_value_seed(RowsSeed {
                        budget: &mut *self.budget,
                    })?);
                }
            }
        }
        if !about {
            return Err(serde::de::Error::missing_field("about"));
        }
        Ok((
            schema.ok_or_else(|| serde::de::Error::missing_field("schema"))?,
            anchors.ok_or_else(|| serde::de::Error::missing_field("anchors"))?,
        ))
    }
}
pub(crate) fn parse_table(
    json: &str,
    budget: &mut SourceRequestBudget<'_>,
) -> std::result::Result<(u32, Vec<crate::evidence::AnchorRow>), String> {
    let mut d = serde_json::Deserializer::from_str(json);
    let table = TableSeed { budget }
        .deserialize(&mut d)
        .map_err(|e| e.to_string())?;
    d.end().map_err(|e| e.to_string())?;
    Ok(table)
}

/// Exact requested metadata layouts for one SDXL source-context capture.
///
/// Borrows the actual request without allocating or cloning image/media buffers.
/// This is the existing controls-Vec plus Arc ownership law, NOT live headroom,
/// physical/RSS bytes, a model-memory bound, or a capture/execution permit.
/// Callers must still perform `record_source_context` with the same real
/// byte-sensitive capacity callback; this query authorizes no allocation itself.
pub fn source_context_requested_bytes(request: &GenerationRequest) -> Result<u64> {
    capture_requested_bytes(SamplingSourceView::from_request(request))
}

/// Shared law for production queries and the existing zero/boundary/allocation tests.
/// Exact newly owned requested layouts for a one-pass actual capture, not import growth peaks.
pub(crate) fn capture_requested_bytes(view: SamplingSourceView<'_>) -> Result<u64> {
    let vector =
        Layout::array::<mlx_gen_sdxl::sampling::IndexedImageSourceFacts>(view.work().control_count)
            .map_err(|_| invalid("capture Control layout overflow"))?;
    (vector.size() as u64)
        .checked_add(arc_request_layout()?.size() as u64)
        .ok_or_else(|| invalid("capture source layout overflow"))
}

/// Allocation-refusal fallback preserves the existing schema/case/status wire, without cloning
/// the Case string or retaining source facts in a Value. If even this bounded output is refused,
/// the existing String-returning harness cannot emit a line without another allocation.
pub fn serialize_capture_failure(
    case: &serde_json::Value,
    budget: &mut SourceRequestBudget<'_>,
) -> Result<String> {
    #[derive(Serialize)]
    struct Failure<'a> {
        schema: u8,
        case: &'a serde_json::Value,
        status: &'static str,
        production_success: bool,
        reason: &'static str,
    }
    serialize_framed_with_budget(
        "SMOKE: FAIL\nCAMPAIGN_JSON ",
        "\n",
        &Failure {
            schema: 1,
            case,
            status: "diagnostic_failure",
            production_success: false,
            reason: "source metadata output budget refused",
        },
        budget,
    )
}

/// Existing anchor fixtures must carry their own REAL request context under a finite requested
/// allowance. This does not turn synthetic telemetry into execution/physical evidence.
#[cfg(test)]
pub(crate) fn fixture_key(
    registry: &mlx_gen::gen_core::ProviderRegistry,
    candidate: &crate::IosCandidate,
    machine: &str,
) -> crate::evidence::EvidenceKey {
    let mut key = crate::evidence_key(registry, candidate, machine).unwrap();
    if let crate::IosWork::Generate(request) = &candidate.work {
        if candidate.provider == crate::sdxl::SDXL_ID {
            let bytes = capture_requested_bytes(SamplingSourceView::from_request(request)).unwrap();
            record_source_context(
                &mut key,
                candidate,
                &mut SourceRequestBudget::bounded(bytes),
            )
            .unwrap();
        }
    }
    key
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence::{
        AnchorKind, AnchorLoadForm, AnchorOutcome, AnchorRow, AnchorSelection, EvidenceKey,
        RefusedQuantum,
    };
    use mlx_gen::gen_core::{Conditioning, ControlKind, Image};
    use mlx_gen_sdxl::sampling::{IndexedImageSourceFacts, SamplingRecipe};

    fn image(width: u32, height: u32, tag: u8) -> Image {
        Image {
            width,
            height,
            pixels: (0..width as usize * height as usize * 3)
                .map(|i| tag.wrapping_add(i as u8))
                .collect(),
        }
    }
    fn request(case: usize) -> GenerationRequest {
        let reference = || Conditioning::Reference {
            image: image(7, 3, 11),
            strength: Some(0.6),
        };
        let control = |tag| Conditioning::Control {
            image: image(5, 9, tag),
            kind: ControlKind::Canny,
            scale: Some(0.75),
        };
        GenerationRequest {
            prompt: "source λ\n\"prompt\"".into(),
            negative_prompt: Some("negative 雪".into()),
            conditioning: match case {
                0 => vec![],
                1 => vec![reference()],
                2 => vec![
                    reference(),
                    Conditioning::Mask {
                        image: image(3, 8, 29),
                    },
                ],
                3 => vec![control(47), control(83)],
                4 => vec![reference(), control(47), control(83)],
                _ => unreachable!(),
            },
            ..Default::default()
        }
    }
    fn capture(request: &GenerationRequest) -> SdxlWorkBinding {
        let bytes = capture_requested_bytes(SamplingSourceView::from_request(request)).unwrap();
        SdxlWorkBinding::capture(request, &mut SourceRequestBudget::bounded(bytes)).unwrap()
    }
    fn import_allowance(binding: &SdxlWorkBinding) -> u64 {
        let facts = binding.require_facts().unwrap();
        // Import reserve_exact may retain old+new Control buffers; this is a requested-layout
        // fixture allowance, not an input/image cap or a physical allocation estimate.
        binding.retained_requested_bytes().unwrap()
            + Layout::array::<IndexedImageSourceFacts>(facts.controls().len())
                .unwrap()
                .size() as u64
    }
    fn key(request: &GenerationRequest) -> EvidenceKey {
        let work = SamplingSourceView::from_request(request).work();
        let token = SamplingRecipe::from_request(request)
            .unwrap()
            .canonical_token();
        EvidenceKey {
            machine: "fixture-machine".into(),
            provider: "sdxl".into(),
            route: "sdxl".into(),
            tier: "q8".into(),
            layout_sha256: "ab".repeat(32),
            architecture_sha256: "cd".repeat(32),
            configured_variant_sha256: (work.control_count > 1).then(|| "ef".repeat(32)),
            mode: if work.control_count > 0 {
                "control"
            } else if work.mask_count > 0 {
                "edit"
            } else if work.reference_count > 0 {
                "image_to_image"
            } else {
                "text_to_image"
            }
            .into(),
            width: request.width,
            height: request.height,
            batch: request.count,
            reference_count: (work.reference_count + work.mask_count + work.control_count) as u32,
            mask_count: work.mask_count as u32,
            control_count: work.control_count as u32,
            load_form: AnchorLoadForm::ResidentEager,
            execution: format!("Resident|EagerMaterialization|fixture|recipe={token}"),
            sampling_recipe: Some(token),
            sdxl_work: Some(SdxlWorkBinding::query(request)),
            token_window: 77,
        }
    }
    fn import_binding(
        json: &str,
        limit: u64,
    ) -> std::result::Result<SdxlWorkBinding, serde_json::Error> {
        let mut d = serde_json::Deserializer::from_str(json);
        let value = BindingSeed {
            budget: &mut SourceRequestBudget::bounded(limit),
        }
        .deserialize(&mut d)?;
        d.end()?;
        Ok(value)
    }

    #[test]
    fn legacy_wire_is_structural_not_complete_source_replay() {
        let r = request(0);
        let mut legacy = key(&r);
        legacy.sdxl_work = None;
        let wire = serde_json::to_string(&legacy).unwrap();
        assert!(!wire.contains("sdxl_work"));
        let decoded: EvidenceKey = serde_json::from_str(&wire).unwrap();
        assert_eq!(decoded, legacy);
        assert!(decoded.family().is_ok());
        let query = SdxlWorkBinding::query(&r);
        assert!(
            query.require_facts().is_err(),
            "digest/token window cannot recover even TEXT UTF8 basis"
        );
        let facts = capture(&r);
        assert_eq!(
            facts.require_facts().unwrap().text().prompt_utf8_bytes,
            r.prompt.len() as u64
        );
    }

    #[test]
    fn source_keys_are_exact_while_families_stay_structural() {
        let a = request(4);
        let mut b = a.clone();
        if let Conditioning::Reference { image, .. } = &mut b.conditioning[0] {
            image.width = 2;
            image.height = 11;
        }
        if let Conditioning::Control { image, .. } = &mut b.conditioning[2] {
            image.pixels.push(19);
        }
        b.prompt.push_str("    ");
        b.negative_prompt = Some("a longer negative λ".into());
        let (_, query_allocated) =
            crate::binding::tests::allocations_during(|| SdxlWorkBinding::query(&a));
        assert_eq!(
            query_allocated, 0,
            "only the NEW scalar source projection/digest, not evidence_key/tokenization"
        );
        let ka = key(&a);
        let kb = key(&b);
        assert_ne!(ka, kb);
        assert_eq!(ka.family().unwrap(), kb.family().unwrap());
        assert_eq!(ka.layout_sha256, kb.layout_sha256);
        assert_eq!(ka.token_window, kb.token_window);
        let before_order = SdxlWorkBinding::query(&b);
        b.conditioning.swap(1, 2);
        assert_ne!(
            before_order,
            SdxlWorkBinding::query(&b),
            "ordered Controls carry their own distinct spans"
        );
        b.conditioning.swap(1, 2);
        // Declared key windows are deliberately held equal; this test does not run a tokenizer.
        let text = capture(&b);
        let text = text.require_facts().unwrap().text();
        assert_eq!(text.used_negative(false), None);
        assert_eq!(
            text.used_negative(true),
            Some(b.negative_prompt.as_ref().unwrap().len() as u64)
        );
    }

    #[test]
    fn public_source_context_query_borrows_every_role_without_allocating() {
        for case in 0..=4 {
            for large in [false, true] {
                let mut actual = request(case);
                if large {
                    actual.prompt = "large λ\n\"source\"".repeat(4096);
                    actual.negative_prompt = Some("negative 雪\n\"escaped\"".repeat(4096));
                    for role in &mut actual.conditioning {
                        let image = match role {
                            Conditioning::Reference { image, .. }
                            | Conditioning::Mask { image }
                            | Conditioning::Control { image, .. } => image,
                            _ => unreachable!(),
                        };
                        image.width = 512;
                        image.height = 512;
                        image.pixels.resize(512 * 512 * 3, 17);
                    }
                }
                let expected =
                    capture_requested_bytes(SamplingSourceView::from_request(&actual)).unwrap();
                let (queried, allocated) = crate::binding::tests::allocations_during(|| {
                    source_context_requested_bytes(&actual)
                });
                assert_eq!(queried.unwrap(), expected);
                assert_eq!(
                    allocated, 0,
                    "metadata query must not clone images or allocate owners"
                );
            }
        }
    }

    #[test]
    fn full_roles_roundtrip_and_key_clones_share_verified_facts() {
        for case in 1..=4 {
            let r = request(case);
            let buffers = |r: &GenerationRequest| {
                r.conditioning
                    .iter()
                    .map(|role| {
                        let image = match role {
                            Conditioning::Reference { image, .. }
                            | Conditioning::Mask { image }
                            | Conditioning::Control { image, .. } => image,
                            _ => unreachable!(),
                        };
                        (
                            image.pixels.as_ptr(),
                            image.pixels.len(),
                            image.pixels.capacity(),
                        )
                    })
                    .collect::<Vec<_>>()
            };
            let before = buffers(&r);
            let requested = source_context_requested_bytes(&r).unwrap();
            let mut capture_budget = SourceRequestBudget::bounded(requested);
            let (owned, allocated) = crate::binding::tests::allocations_during(|| {
                SdxlWorkBinding::capture(&r, &mut capture_budget)
            });
            let owned = owned.unwrap();
            assert_eq!(allocated as u64,requested,"only the exact facts Vec and Arc requested layouts may be allocated; not a physical/RSS assertion");
            assert_eq!(buffers(&r), before);
            let query = SdxlWorkBinding::query(&r);
            assert_eq!(owned, query);
            owned.validate_actual(&r).unwrap();
            let wire = serde_json::to_string(&owned).unwrap();
            let replay = import_binding(&wire, import_allowance(&owned)).unwrap();
            assert_eq!(replay, owned);
            replay.validate_actual(&r).unwrap();
            let f = replay.require_facts().unwrap();
            assert_eq!(f.work(), SamplingSourceView::from_request(&r).work());
            assert_eq!(
                f.controls()
                    .iter()
                    .map(|c| c.source_index)
                    .collect::<Vec<_>>(),
                SamplingSourceView::from_request(&r)
                    .controls()
                    .map(|c| c.source_index)
                    .collect::<Vec<_>>()
            );
            let mut k = key(&r);
            k.sdxl_work = Some(owned);
            let cloned = k.clone();
            let a = k.sdxl_work.as_ref().unwrap();
            let b = cloned.sdxl_work.as_ref().unwrap();
            assert!(Arc::ptr_eq(
                a.facts.as_ref().unwrap(),
                b.facts.as_ref().unwrap()
            ));
            assert_eq!(
                a.require_facts().unwrap().controls().as_ptr(),
                b.require_facts().unwrap().controls().as_ptr()
            );
        }
    }

    #[test]
    fn ordinary_deserialize_rejects_facts_before_the_sequence() {
        let owned = capture(&request(3));
        let wire = serde_json::to_string(&owned).unwrap();
        assert!(serde_json::from_str::<SdxlWorkBinding>(&wire)
            .unwrap_err()
            .to_string()
            .contains("explicit budgeted import"));
        // Deliberately incomplete sequence: our error must precede any sequence parsing/EOF.
        let poison = format!("{{\"sha256\":\"{:?}\",\"facts\":[", owned.sha256);
        assert!(serde_json::from_str::<SdxlWorkBinding>(&poison)
            .unwrap_err()
            .to_string()
            .contains("explicit budgeted import"));
        let mut k = key(&request(3));
        k.sdxl_work = Some(owned.clone());
        let key_wire = serde_json::to_string(&k).unwrap();
        assert!(serde_json::from_str::<EvidenceKey>(&key_wire).is_err());
        assert!(serde_json::from_value::<EvidenceKey>(serde_json::to_value(&k).unwrap()).is_err());
        let mut d = serde_json::Deserializer::from_str(&key_wire);
        let mut budget = SourceRequestBudget::bounded(import_allowance(&owned));
        assert_eq!(
            KeySeed {
                budget: &mut budget
            }
            .deserialize(&mut d)
            .unwrap(),
            k
        );
        d.end().unwrap();
        let mut forged = serde_json::to_value(&owned).unwrap();
        forged["facts"]["controls"][1]["source_index"] =
            forged["facts"]["controls"][0]["source_index"].clone();
        assert!(import_binding(&forged.to_string(), import_allowance(&owned)).is_err());
        let mut outside = serde_json::to_value(&owned).unwrap();
        outside["facts"]["controls"][1]["source_index"] = u64::MAX.into();
        let raw_wire = outside["facts"].to_string();
        let mut raw_d = serde_json::Deserializer::from_str(&raw_wire);
        let mut raw_budget = SourceRequestBudget::bounded(import_allowance(&owned));
        let raw = SamplingSourceFactsSeed {
            budget: &mut raw_budget,
        }
        .deserialize(&mut raw_d)
        .unwrap();
        outside["sha256"] = serde_json::to_value(digest(raw.view())).unwrap();
        assert!(
            import_binding(&outside.to_string(), import_allowance(&owned))
                .unwrap_err()
                .to_string()
                .contains("outside"),
            "a recomputed hash does not authorize nonexistent source occurrences"
        );
        let mut missing = serde_json::to_value(&owned).unwrap();
        missing["facts"].as_object_mut().unwrap().remove("text");
        assert!(import_binding(&missing.to_string(), import_allowance(&owned)).is_err());
        let mut unknown = serde_json::to_value(&owned).unwrap();
        unknown["facts"]["extra"] = true.into();
        assert!(import_binding(&unknown.to_string(), import_allowance(&owned)).is_err());
    }

    #[test]
    fn source_facts_do_not_replace_full_content_or_validate_media_early() {
        let a = request(2);
        let mut b = a.clone();
        if let Conditioning::Reference { image, .. } = &mut b.conditioning[0] {
            image.pixels[0] ^= 1;
        }
        assert_eq!(SdxlWorkBinding::query(&a), SdxlWorkBinding::query(&b));
        assert_ne!(
            crate::binding::generation_binding(&a),
            crate::binding::generation_binding(&b)
        );
        b.negative_prompt = Some("negative 水".into()); // same UTF8 span, different content
        assert_eq!(SdxlWorkBinding::query(&a), SdxlWorkBinding::query(&b));
        assert_ne!(
            crate::binding::generation_binding(&a),
            crate::binding::generation_binding(&b)
        );
        let mut malformed = request(4);
        if let Conditioning::Reference { image, .. } = &mut malformed.conditioning[0] {
            image.width = 0;
            image.pixels.pop();
        }
        if let Conditioning::Control { image, .. } = &mut malformed.conditioning[1] {
            image.height = 0;
            image.pixels.clear();
        }
        let raw = capture(&malformed);
        raw.validate_actual(&malformed).unwrap();
        assert_eq!(
            raw.require_facts()
                .unwrap()
                .work()
                .reference_image
                .unwrap()
                .width,
            0
        );
        let Conditioning::Reference { image, .. } = &malformed.conditioning[0] else {
            unreachable!()
        };
        assert_eq!(
            raw.require_facts()
                .unwrap()
                .work()
                .reference_image
                .unwrap()
                .buffer_bytes,
            image.pixels.len() as u64,
            "actual retained byte span is not width*height*3"
        );
        assert_eq!(
            raw.require_facts().unwrap().controls()[0]
                .image
                .buffer_bytes,
            0
        );
        // Original image/Mask/Control preprocessing still owns those malformed-input errors.
    }

    #[test]
    fn requested_budget_and_typed_capture_import_pipeline_are_connected() {
        use std::cell::RefCell;
        let r = request(4);
        let events = RefCell::new(Vec::new());
        let mut next = |bytes| {
            events.borrow_mut().push(bytes);
            Ok(())
        };
        let mut denied = SourceRequestBudget::with_capacity(0, &mut next);
        assert!(SdxlWorkBinding::capture(&r, &mut denied).is_err());
        assert_eq!(denied.held_requested_bytes(), 0);
        assert!(events.borrow().is_empty());
        let mut error = Some(Error::Unsupported("fixture next allocation refused".into()));
        let mut refuse = |bytes| {
            assert!(bytes > 0);
            Err(error.take().expect("first request must stop"))
        };
        let allowance = source_context_requested_bytes(&r).unwrap();
        let mut live_denied = SourceRequestBudget::with_capacity(allowance, &mut refuse);
        let (refused, allocated) = crate::binding::tests::allocations_during(|| {
            SdxlWorkBinding::capture(&r, &mut live_denied)
        });
        assert!(refused.is_err());
        assert_eq!(
            allocated, 0,
            "preallocated fixture error: callback must precede the source Vec allocation"
        );
        assert_eq!(live_denied.held_requested_bytes(), 0);
        assert!(Layout::array::<IndexedImageSourceFacts>(usize::MAX).is_err());
        let mut overflow = SourceRequestBudget::bounded(u64::MAX);
        overflow.account_retained(u64::MAX).unwrap();
        assert!(overflow.before_replace(0, 1).is_err());
        let binding = capture(&r);
        let mut k = key(&r);
        k.sdxl_work = Some(binding.clone());
        let row = serde_json::json!({"schema":1,"status":"diagnostic_success","production_success":false});
        assert!(row.get("evidence_key").is_none());
        let mut expected = row.clone();
        expected["evidence_key"] = serde_json::to_value(&k).unwrap();
        let prefix = "SMOKE: FAIL\nCAMPAIGN_JSON ";
        let exact = prefix.len() + serde_json::to_string(&expected).unwrap().len() + 1;
        let retained = binding.retained_requested_bytes().unwrap();
        events.borrow_mut().clear();
        events.borrow_mut().reserve_exact(1); // instrumentation storage outside the observation
        let mut output_budget =
            SourceRequestBudget::with_capacity(retained + exact as u64, &mut next);
        output_budget.account_retained(retained).unwrap();
        let (output, allocated) = crate::binding::tests::allocations_during(|| {
            serialize_capture_row(prefix, &row, Some(&k), &mut output_budget)
        });
        let output = output.unwrap();
        assert_eq!(allocated,exact,"only the once-reserved output buffer, no intermediate Value/facts copy; requested Rust bytes only");
        assert_eq!(output.len(), exact);
        assert_eq!(&*events.borrow(), &[exact as u64]);
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(output.strip_prefix(prefix).unwrap())
                .unwrap(),
            expected
        );
        assert!(
            row.get("evidence_key").is_none(),
            "typed facts were not inserted into the Value row"
        );
        let anchor = AnchorRow {
            key: k.clone(),
            anchor_kind: AnchorKind::Resident,
            selection: AnchorSelection::default(),
            outcome: AnchorOutcome::Refused {
                refused_quantum: RefusedQuantum {
                    label: "fixture".into(),
                    scratch_bytes: 1,
                    reserve_bytes: 0,
                    available_bytes: Some(0),
                },
                run_id: "fixture-run".into(),
                build_sha: "ab".repeat(20),
            },
            evidence_class: "device_capture".into(),
        };
        let table =
            serde_json::json!({"schema":3,"about":"explicit finite fixture","anchors":[anchor]})
                .to_string();
        let mut input_budget = SourceRequestBudget::bounded(table.len() as u64);
        let (schema, rows) = parse_table(&table, &mut input_budget).unwrap();
        assert_eq!(schema, 3);
        assert_eq!(rows[0].key, k);
        assert!(
            serde_json::from_str::<AnchorRow>(&serde_json::to_string(&rows[0]).unwrap()).is_err()
        );
        let mut short = SourceRequestBudget::bounded(retained + exact as u64 - 1);
        short.account_retained(retained).unwrap();
        assert!(serialize_capture_row(prefix, &row, Some(&k), &mut short).is_err());
    }
}
