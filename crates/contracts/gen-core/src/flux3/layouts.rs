//! The FLUX 3 layout document and resolver-owned reference bindings a
//! [`GenerationRequest`](crate::GenerationRequest) carries. Layout compilation lives in
//! `gen_core_flux3::layouts` (sc-27316).

use super::pup;
use crate::{Error, Result};
use serde_json::Value;

fn invalid(message: impl Into<String>) -> Error {
    Error::Unsupported(format!("flux3 layout: {}", message.into()))
}
/// Original authoring document. Construct through validation; never replace it with locked rows.
#[derive(Clone, Debug, PartialEq)]
pub struct Layout(Value);
/// Resolver-owned binding in actual attachment order. Never accept these dimensions/orientation
/// as untrusted memory hints. EXIF orientation follows TIFF's 1..=8 convention.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReferenceBinding {
    pub id: String,
    pub original_size: (u32, u32),
    pub orientation: u8,
}
impl ReferenceBinding {
    pub fn oriented(id: impl Into<String>, size: (u32, u32)) -> Self {
        Self {
            id: id.into(),
            original_size: size,
            orientation: 1,
        }
    }

    pub fn oriented_size(&self) -> (u32, u32) {
        if self.orientation >= 5 {
            (self.original_size.1, self.original_size.0)
        } else {
            self.original_size
        }
    }
}
impl Layout {
    pub fn parse(json: &str) -> Result<Self> {
        Self::from_json(
            serde_json::from_str(json).map_err(|e| invalid(format!("invalid JSON: {e}")))?,
        )
    }

    pub fn from_json(value: Value) -> Result<Self> {
        let object = value
            .as_object()
            .ok_or_else(|| invalid("expected a layout document"))?;
        if object.get("version").and_then(Value::as_u64) != Some(1) {
            return Err(invalid("unsupported layout version; expected 1"));
        }
        let rows = object
            .get("boxes")
            .and_then(Value::as_array)
            .ok_or_else(|| invalid("boxes must be a list"))?;
        let mut ids = std::collections::HashSet::new();
        for (i, row) in rows.iter().enumerate() {
            let row = row
                .as_object()
                .ok_or_else(|| invalid(format!("box {i} must be an object")))?;
            let id = row
                .get("id")
                .and_then(Value::as_str)
                .ok_or_else(|| invalid(format!("box {i} needs a stable id")))?;
            if id.is_empty() || !ids.insert(id.to_owned()) {
                return Err(invalid(format!("box {i} id must be nonempty and distinct")));
            }
            if row.get("desc").is_some_and(|v| !v.is_string()) {
                return Err(invalid(format!("box {id} desc must be text")));
            }
            for key in ["units", "bbox_units", "src_units", "tgt_units"] {
                if row.contains_key(key) {
                    return Err(invalid(format!(
                        "box {id}: {key} is unsupported; use an explicit coordinate space"
                    )));
                }
            }
            let bbox = coords(row.get("bbox"))?;
            let src = coords(row.get("src_bbox"))?;
            let tgt = coords(row.get("tgt_bbox"))?;
            for (key, present) in [
                ("bbox_space", bbox.is_some()),
                ("src_space", src.is_some()),
                ("tgt_space", tgt.is_some()),
            ] {
                if row.contains_key(key) && !present {
                    return Err(invalid(format!(
                        "box {id}: {key} has no corresponding coordinates"
                    )));
                }
            }
            if bbox.is_some() {
                if row.contains_key("src_bbox")
                    || row.contains_key("tgt_bbox")
                    || row.contains_key("from")
                    || row.contains_key("reference_id")
                {
                    return Err(invalid(format!("box {id} mixes placement and edit fields")));
                }
                space(row.get("bbox_space"))?;
            } else {
                if src.is_none() && tgt.is_none() {
                    return Err(invalid(format!("box {id} needs bbox or src_bbox/tgt_bbox")));
                }
                let origin = row
                    .get("from")
                    .and_then(Value::as_str)
                    .ok_or_else(|| invalid(format!("box {id} needs from")))?;
                if !pup::is_row_origin(origin) || (origin == "target") == src.is_some() {
                    return Err(invalid(format!(
                        "box {id}: target placement requires null/absent source; reference edit requires source"
                    )));
                }
                if origin == "target" && row.contains_key("reference_id") {
                    return Err(invalid(format!(
                        "box {id}: target cannot select a reference"
                    )));
                }
                if row
                    .get("reference_id")
                    .is_some_and(|v| !v.is_string() || v.as_str() == Some(""))
                {
                    return Err(invalid(format!(
                        "box {id}: reference_id must be nonempty text"
                    )));
                }
                space(row.get("src_space"))?;
                space(row.get("tgt_space"))?;
            }
        }
        Ok(Self(value))
    }

    /// Includes unknown provenance fields unchanged. Do not save the compiled table in its place.
    pub fn to_json(&self) -> &Value {
        &self.0
    }
}

// Authoring-space parsing shared with `gen_core_flux3::layouts::LayoutCompile` (sc-27316).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[doc(hidden)]
pub enum Space {
    Normalized,
    Original,
    Oriented,
    Fitted,
    ReferenceCanvas,
    OutputCanvas,
    Output,
}
#[doc(hidden)]
pub fn space(value: Option<&Value>) -> Result<Space> {
    match value {
        None => Ok(Space::Normalized),
        Some(Value::String(s)) => match s.as_str() {
            "normalized" => Ok(Space::Normalized),
            "original" => Ok(Space::Original),
            "oriented" => Ok(Space::Oriented),
            "fitted" => Ok(Space::Fitted),
            "reference_canvas" => Ok(Space::ReferenceCanvas),
            "output_canvas" => Ok(Space::OutputCanvas),
            "output" => Ok(Space::Output),
            _ => Err(invalid(format!("unsupported coordinate space {s:?}"))),
        },
        _ => Err(invalid(
            "coordinate space must be text (absence means normalized)",
        )),
    }
}
#[doc(hidden)]
pub fn coords(value: Option<&Value>) -> Result<Option<[f64; 4]>> {
    let Some(value) = value.filter(|v| !v.is_null()) else {
        return Ok(None);
    };
    let list = value
        .as_array()
        .filter(|a| a.len() == 4)
        .ok_or_else(|| invalid("box must have four numeric YXYX coordinates"))?;
    let mut out = [0.0; 4];
    for (i, v) in list.iter().enumerate() {
        out[i] = v
            .as_f64()
            .filter(|v| v.is_finite())
            .ok_or_else(|| invalid("box coordinates must be finite numbers, not booleans"))?;
    }
    if out[2] <= out[0] || out[3] <= out[1] {
        return Err(invalid("box must have positive height and width"));
    }
    // Finite outside coordinates are valid. bbox.py clamps ONLY the derived 0..1000 table.
    Ok(Some(out))
}
