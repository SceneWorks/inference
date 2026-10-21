//! SDXL-only payload-free source context. This records requested layouts, not physical memory.
//! No image/text validation, tokenization, recipe normalization, or native work is performed here.
use super::{ReferenceImageFacts, SamplingWorkFacts};
use mlx_gen::gen_core::{Conditioning, GenerationRequest};
use mlx_gen::{Error, Result};
use serde::de::{DeserializeSeed, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use std::alloc::Layout;
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SamplingTextSourceFacts {
    pub prompt_utf8_bytes: u64,
    pub negative_prompt_utf8_bytes: u64,
}
impl SamplingTextSourceFacts {
    pub fn used_negative(self, cfg_on: bool) -> Option<u64> {
        cfg_on.then_some(self.negative_prompt_utf8_bytes)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IndexedImageSourceFacts {
    pub source_index: u64,
    pub image: ReferenceImageFacts,
}

// Deliberately neither Clone nor Deserialize: unbounded ownership has exactly two budgeted
// constructors below. Catalog keys share an immutable Arc instead of cloning this Vec.
#[derive(Serialize)]
pub struct SamplingSourceFacts {
    work: SamplingWorkFacts,
    text: SamplingTextSourceFacts,
    reference_index: Option<u64>,
    mask: Option<IndexedImageSourceFacts>,
    controls: Vec<IndexedImageSourceFacts>,
}
impl SamplingSourceFacts {
    pub fn view(&self) -> SamplingSourceView<'_> {
        SamplingSourceView::Recorded(self)
    }
    pub fn work(&self) -> SamplingWorkFacts {
        self.work
    }
    pub fn text(&self) -> SamplingTextSourceFacts {
        self.text
    }
    pub fn reference_index(&self) -> Option<u64> {
        self.reference_index
    }
    pub fn mask(&self) -> Option<IndexedImageSourceFacts> {
        self.mask
    }
    pub fn controls(&self) -> &[IndexedImageSourceFacts] {
        &self.controls
    }

    pub fn capture(
        view: SamplingSourceView<'_>,
        budget: &mut SourceRequestBudget<'_>,
    ) -> Result<Self> {
        let mut controls = Vec::new();
        let count = view.work().control_count;
        reserve_controls(&mut controls, count, budget)?;
        for item in view.controls() {
            // A recorded raw view may have inconsistent claimed counts. Never let Vec::extend
            // turn that into an unbudgeted amortized growth before coherence is checked.
            let needed = controls
                .len()
                .checked_add(1)
                .ok_or_else(|| invalid("Control count overflow"))?;
            reserve_controls(&mut controls, needed, budget)?;
            controls.push(item);
        }
        Ok(Self {
            work: view.work(),
            text: view.text(),
            reference_index: view.reference_index(),
            mask: view.mask(),
            controls,
        })
    }

    /// Context coherence only. Zero dimensions and non-RGB spans remain raw facts; the original
    /// provider preprocessing owns their errors at its original point in execution.
    pub fn validate_roles(&self) -> Result<()> {
        let reference_ok = match self.work.reference_count {
            0 => self.reference_index.is_none() && self.work.reference_image.is_none(),
            1 => self.reference_index.is_some() && self.work.reference_image.is_some(),
            _ => false, // the existing SDXL resolver accepts at most one Reference
        };
        if !reference_ok
            || self.controls.len() != self.work.control_count
            || self.work.mask_count != usize::from(self.mask.is_some())
        {
            return Err(invalid("source context disagrees with resolved SDXL roles"));
        }
        let total = self
            .work
            .reference_count
            .checked_add(self.work.mask_count)
            .and_then(|n| n.checked_add(self.work.control_count))
            .ok_or_else(|| invalid("source role count overflow"))? as u64;
        if self.reference_index.is_some_and(|i| i >= total)
            || self.mask.is_some_and(|m| m.source_index >= total)
        {
            return Err(invalid(
                "source occurrence is outside its closed role sequence",
            ));
        }
        let mut previous = None;
        for control in &self.controls {
            if control.source_index >= total {
                return Err(invalid(
                    "source occurrence is outside its closed role sequence",
                ));
            }
            if previous.is_some_and(|p| p >= control.source_index)
                || self.reference_index == Some(control.source_index)
                || self
                    .mask
                    .is_some_and(|m| m.source_index == control.source_index)
            {
                return Err(invalid(
                    "source occurrence indices are duplicated or unordered",
                ));
            }
            previous = Some(control.source_index);
        }
        if self.reference_index.is_some()
            && self
                .mask
                .is_some_and(|m| Some(m.source_index) == self.reference_index)
        {
            return Err(invalid("Reference and Mask source indices collide"));
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub enum SamplingSourceView<'a> {
    Actual(&'a GenerationRequest),
    Recorded(&'a SamplingSourceFacts),
}
impl<'a> SamplingSourceView<'a> {
    pub fn from_request(request: &'a GenerationRequest) -> Self {
        Self::Actual(request)
    }
    pub fn work(self) -> SamplingWorkFacts {
        match self {
            Self::Actual(r) => SamplingWorkFacts::from_request(r),
            Self::Recorded(f) => f.work,
        }
    }
    pub fn text(self) -> SamplingTextSourceFacts {
        match self {
            Self::Actual(r) => SamplingTextSourceFacts {
                prompt_utf8_bytes: r.prompt.len() as u64,
                negative_prompt_utf8_bytes: r.negative_prompt.as_deref().unwrap_or("").len() as u64,
            },
            Self::Recorded(f) => f.text,
        }
    }
    pub fn reference_index(self) -> Option<u64> {
        match self {
            Self::Actual(r) => r
                .conditioning
                .iter()
                .position(|c| matches!(c, Conditioning::Reference { .. }))
                .map(|i| i as u64),
            Self::Recorded(f) => f.reference_index,
        }
    }
    pub fn mask(self) -> Option<IndexedImageSourceFacts> {
        match self {
            Self::Actual(r) => r
                .conditioning
                .iter()
                .enumerate()
                .find_map(|(i, c)| match c {
                    Conditioning::Mask { image } => Some(IndexedImageSourceFacts {
                        source_index: i as u64,
                        image: ReferenceImageFacts::from_image(image),
                    }),
                    _ => None,
                }),
            Self::Recorded(f) => f.mask,
        }
    }
    pub fn controls(self) -> ControlSources<'a> {
        match self {
            Self::Actual(r) => ControlSources::Actual(r.conditioning.iter().enumerate()),
            Self::Recorded(f) => ControlSources::Recorded(f.controls.iter()),
        }
    }
}
pub enum ControlSources<'a> {
    Actual(std::iter::Enumerate<std::slice::Iter<'a, Conditioning>>),
    Recorded(std::slice::Iter<'a, IndexedImageSourceFacts>),
}
impl Iterator for ControlSources<'_> {
    type Item = IndexedImageSourceFacts;
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Actual(iter) => iter.find_map(|(i, c)| match c {
                Conditioning::Control { image, .. } => Some(IndexedImageSourceFacts {
                    source_index: i as u64,
                    image: ReferenceImageFacts::from_image(image),
                }),
                _ => None,
            }),
            Self::Recorded(iter) => iter.next().copied(),
        }
    }
}

/// Finite requested-layout ledger. A separate optional live hook receives NEXT allocation bytes,
/// never the retained-total ledger (which would double-charge the process snapshot).
/// `bounded` authorizes source requests only, e.g. an explicit artifact/fixture allowance; it is
/// NOT a live-capacity check or evidence of physical growth. No Default/unlimited/TLS mode exists.
pub struct SourceRequestBudget<'a> {
    limit: u64,
    held: u64,
    next_capacity: Option<&'a mut dyn FnMut(u64) -> Result<()>>,
}
impl<'a> SourceRequestBudget<'a> {
    pub fn bounded(limit: u64) -> Self {
        Self {
            limit,
            held: 0,
            next_capacity: None,
        }
    }
    pub fn with_capacity(limit: u64, next_capacity: &'a mut dyn FnMut(u64) -> Result<()>) -> Self {
        Self {
            limit,
            held: 0,
            next_capacity: Some(next_capacity),
        }
    }
    pub fn held_requested_bytes(&self) -> u64 {
        self.held
    }
    /// Account for already-owned verified source storage at a later serialization phase.
    /// No live capacity call: that phase's process snapshot already contains these allocations.
    pub fn account_retained(&mut self, bytes: u64) -> Result<()> {
        let held = self
            .held
            .checked_add(bytes)
            .ok_or_else(|| invalid("retained source request overflow"))?;
        if held > self.limit {
            return Err(invalid("retained source requests exceed allowance"));
        }
        self.held = held;
        Ok(())
    }
    pub fn before_replace(&mut self, old: u64, next: u64) -> Result<()> {
        if old > self.held {
            return Err(invalid("source ledger lost its old allocation"));
        }
        let peak = self
            .held
            .checked_add(next)
            .ok_or_else(|| invalid("source layout peak overflow"))?;
        if peak > self.limit {
            return Err(invalid("source requested-layout allowance exhausted"));
        }
        if next != 0 {
            if let Some(check) = self.next_capacity.as_mut() {
                check(next)?;
            }
        }
        Ok(())
    }
    pub fn commit_replace(&mut self, old: u64, next: u64) {
        // Matching before_replace has checked subtraction and the stricter old+new peak.
        self.held = self.held - old + next;
    }
}
fn invalid(message: &str) -> Error {
    Error::Msg(format!("SDXL source context: {message}"))
}
fn reserve_controls(
    out: &mut Vec<IndexedImageSourceFacts>,
    needed: usize,
    budget: &mut SourceRequestBudget<'_>,
) -> Result<()> {
    if needed <= out.capacity() {
        return Ok(());
    }
    let old = Layout::array::<IndexedImageSourceFacts>(out.capacity())
        .map_err(|_| invalid("old Control layout overflow"))?
        .size() as u64;
    let next = Layout::array::<IndexedImageSourceFacts>(needed)
        .map_err(|_| invalid("Control source layout overflow"))?
        .size() as u64;
    budget.before_replace(old, next)?;
    out.try_reserve_exact(
        needed
            .checked_sub(out.len())
            .ok_or_else(|| invalid("Control count underflow"))?,
    )
    .map_err(|_| invalid("Control source reservation failed"))?;
    budget.commit_replace(old, next);
    Ok(())
}

/// The only full-payload deserializer; each growth is authorized before reserve, independent of
/// an untrusted sequence size_hint. There is intentionally no Deserialize for SamplingSourceFacts.
pub struct SamplingSourceFactsSeed<'a, 'b> {
    pub budget: &'a mut SourceRequestBudget<'b>,
}
impl<'de> DeserializeSeed<'de> for SamplingSourceFactsSeed<'_, '_> {
    type Value = SamplingSourceFacts;
    fn deserialize<D: Deserializer<'de>>(self, d: D) -> std::result::Result<Self::Value, D::Error> {
        d.deserialize_struct(
            "SamplingSourceFacts",
            &["work", "text", "reference_index", "mask", "controls"],
            self,
        )
    }
}
impl<'de> Visitor<'de> for SamplingSourceFactsSeed<'_, '_> {
    type Value = SamplingSourceFacts;
    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("budgeted SDXL source context")
    }
    fn visit_map<M: MapAccess<'de>>(
        self,
        mut map: M,
    ) -> std::result::Result<Self::Value, M::Error> {
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "snake_case")]
        enum Field {
            Work,
            Text,
            ReferenceIndex,
            Mask,
            Controls,
        }
        let (mut work, mut text, mut reference_index, mut mask, mut controls) =
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
                Field::Work => scalar!(work, "work"),
                Field::Text => scalar!(text, "text"),
                Field::ReferenceIndex => scalar!(reference_index, "reference_index"),
                Field::Mask => scalar!(mask, "mask"),
                Field::Controls => {
                    if controls.is_some() {
                        return Err(serde::de::Error::duplicate_field("controls"));
                    }
                    controls = Some(map.next_value_seed(ControlsSeed {
                        budget: &mut *self.budget,
                    })?);
                }
            }
        }
        Ok(SamplingSourceFacts {
            work: work.ok_or_else(|| serde::de::Error::missing_field("work"))?,
            text: text.ok_or_else(|| serde::de::Error::missing_field("text"))?,
            reference_index: reference_index
                .ok_or_else(|| serde::de::Error::missing_field("reference_index"))?,
            mask: mask.ok_or_else(|| serde::de::Error::missing_field("mask"))?,
            controls: controls.ok_or_else(|| serde::de::Error::missing_field("controls"))?,
        })
    }
}
struct ControlsSeed<'a, 'b> {
    budget: &'a mut SourceRequestBudget<'b>,
}
impl<'de> DeserializeSeed<'de> for ControlsSeed<'_, '_> {
    type Value = Vec<IndexedImageSourceFacts>;
    fn deserialize<D: Deserializer<'de>>(self, d: D) -> std::result::Result<Self::Value, D::Error> {
        d.deserialize_seq(self)
    }
}
impl<'de> Visitor<'de> for ControlsSeed<'_, '_> {
    type Value = Vec<IndexedImageSourceFacts>;
    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("ordered source occurrences")
    }
    fn visit_seq<S: SeqAccess<'de>>(
        self,
        mut seq: S,
    ) -> std::result::Result<Self::Value, S::Error> {
        let mut out = Vec::new();
        while let Some(item) = seq.next_element::<IndexedImageSourceFacts>()? {
            let needed = out
                .len()
                .checked_add(1)
                .ok_or_else(|| serde::de::Error::custom("Control count overflow"))?;
            reserve_controls(&mut out, needed, self.budget).map_err(serde::de::Error::custom)?;
            out.push(item);
        }
        Ok(out)
    }
}
