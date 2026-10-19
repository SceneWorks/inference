//! Lossless layout authoring and the frozen BFL `bbox.py` conditioning boundary.
//!
//! Boxes are caption annotations, NOT pixel masks, AE crops or changes to reference token IDs.
//! Keep the author's document separate from the derived 0–1000 locked table. `generate.py`
//! appends that table to the caption (or passes it through existing PUP reconciliation), before
//! deriving a seed or text encoding. Reference images still follow `references::ReferencePlan`.
//!
//! Wire document: `{version:1, boxes:[upstream rows]}`. Upstream normalized YXYX coordinates
//! remain the default. Optional `bbox_space`, `src_space`, `tgt_space` specify explicit pixel
//! authoring spaces: `original`, `oriented`, `fitted`, `reference_canvas`, `output_canvas`,
//! `output` (visible post-crop output). `reference_id` binds a reference by stable caller-owned
//! identity, not a stale attachment index. All original fields, absence, nulls, number precision,
//! IDs and order remain in the original JSON. Only the derived row receives a current ordinal.

use serde_json::Value;

use crate::flux3::{pup, references};
use crate::{Error, Result};

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

/// Derived native input, shared by both backends and the existing PUP implementation.
#[derive(Clone, Debug, PartialEq)]
pub struct CompiledLayout {
    pub boxes: Vec<pup::BoxSpec>,
    pub rows: Vec<pup::JsonValue>,
}

impl CompiledLayout {
    pub fn rows_json(&self) -> String {
        pup::dumps(&pup::JsonValue::Array(self.rows.clone()))
    }

    /// Source `generate.py:63`: bypass skips the VLM, not the locked table. Validate the complete
    /// caption BEFORE seed derivation, text encoding or image work. Never truncate annotations.
    pub fn caption(&self, prompt: &str) -> Result<String> {
        let caption = if self.rows.is_empty() {
            prompt.to_owned()
        } else {
            format!("{} {}", pup::py_rstrip(prompt), self.rows_json())
        };
        pup::validate_caption_size(&caption, "layout caption")?;
        Ok(caption)
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

    /// Validate bindings/geometry and convert explicit pixel authoring spaces, without image work.
    /// A normalized source box is already the frozen upstream input: no invented AE/RoPE warp.
    pub fn compile(
        &self,
        bindings: &[ReferenceBinding],
        plan: Option<&references::ReferencePlan>,
        output: &references::OutputGeometry,
    ) -> Result<CompiledLayout> {
        validate_bindings(bindings, plan)?;
        if output.canvas.0 == 0 || output.canvas.1 == 0 {
            return Err(invalid("empty output canvas"));
        }
        let mut boxes = Vec::new();
        for row in self.0["boxes"].as_array().expect("validated layout") {
            let row = row.as_object().expect("validated row");
            let id = row["id"].as_str().expect("validated id");
            let mut b = pup::BoxSpec {
                id: Some(id.to_owned()),
                desc: row.get("desc").and_then(Value::as_str).map(str::to_owned),
                ..Default::default()
            };
            if let Some(value) = coords(row.get("bbox"))? {
                b.bbox =
                    Some(target_coords(value, space(row.get("bbox_space"))?, output)?.to_vec());
            } else {
                let origin = row["from"].as_str().expect("validated origin");
                if origin == "target" {
                    b.from = Some("target".into());
                } else {
                    let index =
                        if let Some(identity) = row.get("reference_id").and_then(Value::as_str) {
                            bindings
                                .iter()
                                .position(|r| r.id == identity)
                                .ok_or_else(|| {
                                    invalid(format!(
                                        "box {id} selected reference {identity:?} is not attached"
                                    ))
                                })?
                        } else {
                            origin
                                .strip_prefix("ref_image_")
                                .and_then(|n| n.parse::<usize>().ok())
                                .filter(|&n| n < bindings.len())
                                .ok_or_else(|| {
                                    invalid(format!("box {id} references missing {origin}"))
                                })?
                        };
                    b.from = Some(format!("ref_image_{index}"));
                    b.src_bbox = coords(row.get("src_bbox"))?
                        .map(|value| {
                            source_coords(
                                value,
                                space(row.get("src_space"))?,
                                &bindings[index],
                                plan.map(|p| &p.geometry[index]),
                            )
                        })
                        .transpose()?
                        .map(|v| v.to_vec());
                }
                b.tgt_bbox = coords(row.get("tgt_bbox"))?
                    .map(|value| target_coords(value, space(row.get("tgt_space"))?, output))
                    .transpose()?
                    .map(|v| v.to_vec());
            }
            boxes.push(b);
        }
        let rows = pup::to_canvas(&boxes)?;
        let compiled = CompiledLayout { boxes, rows };
        // Even an empty prompt must have room for the FULL locked table, before invoking PUP.
        compiled.caption("")?;
        Ok(compiled)
    }
}

/// Compile against the actual ordered, decoded request before admission or native inference.
/// Callers without stable identities can use upstream ordinal boxes in oriented/normalized space.
/// Stable-ID boxes and original-pixel coordinates require explicit resolver-owned bindings.
pub fn compile_request(req: &crate::GenerationRequest) -> Result<Option<CompiledLayout>> {
    let Some(layout) = &req.flux3_layout else {
        if !req.flux3_layout_bindings.is_empty() {
            return Err(invalid("reference bindings require a layout document"));
        }
        return Ok(None);
    };
    let plan = references::image_request_plan(req)?;
    let fallback;
    let bindings = if req.flux3_layout_bindings.is_empty() {
        for row in layout.to_json()["boxes"]
            .as_array()
            .expect("validated layout")
        {
            if row.get("reference_id").is_some() || row["src_space"] == "original" {
                return Err(invalid(
                    "stable reference identity/original pixels require resolved bindings",
                ));
            }
        }
        fallback = plan
            .as_ref()
            .map(|p| {
                p.geometry
                    .iter()
                    .enumerate()
                    .map(|(i, g)| {
                        ReferenceBinding::oriented(format!("ref_image_{i}"), g.source_size)
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        &fallback
    } else {
        &req.flux3_layout_bindings
    };
    let output = plan
        .as_ref()
        .map(|p| p.output.clone())
        .unwrap_or(references::OutputGeometry {
            canvas: (req.width, req.height),
            crop: None,
        });
    let compiled = layout.compile(bindings, plan.as_ref(), &output)?;
    compiled.caption(&req.prompt)?;
    Ok(Some(compiled))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Space {
    Normalized,
    Original,
    Oriented,
    Fitted,
    ReferenceCanvas,
    OutputCanvas,
    Output,
}

fn space(value: Option<&Value>) -> Result<Space> {
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

fn coords(value: Option<&Value>) -> Result<Option<[f64; 4]>> {
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

fn validate_bindings(
    bindings: &[ReferenceBinding],
    plan: Option<&references::ReferencePlan>,
) -> Result<()> {
    if bindings.len() > references::MAX_REFERENCES
        || plan.is_some_and(|p| {
            p.geometry.len() != bindings.len() || p.canvases.len() != bindings.len()
        })
        || (!bindings.is_empty() && plan.is_none())
    {
        return Err(invalid(
            "ordered bindings must match the actual reference plan",
        ));
    }
    let mut ids = std::collections::HashSet::new();
    for (i, b) in bindings.iter().enumerate() {
        if b.id.is_empty()
            || !ids.insert(&b.id)
            || !(1..=8).contains(&b.orientation)
            || b.original_size.0 == 0
            || b.original_size.1 == 0
        {
            return Err(invalid(
                "reference bindings need distinct identities and valid decoded geometry",
            ));
        }
        if plan.is_some_and(|p| p.geometry[i].source_size != b.oriented_size()) {
            return Err(invalid(
                "decoded EXIF-oriented dimensions disagree with reference plan",
            ));
        }
    }
    Ok(())
}

fn normalize(v: [f64; 4], size: (u32, u32)) -> Result<[f64; 4]> {
    if size.0 == 0 || size.1 == 0 {
        return Err(invalid("empty coordinate frame"));
    }
    let v = [
        v[0] / size.1 as f64,
        v[1] / size.0 as f64,
        v[2] / size.1 as f64,
        v[3] / size.0 as f64,
    ];
    if v.iter().any(|v| !v.is_finite()) {
        return Err(invalid("coordinate transform overflow"));
    }
    Ok(v)
}

fn target_coords(
    mut v: [f64; 4],
    space: Space,
    output: &references::OutputGeometry,
) -> Result<[f64; 4]> {
    match space {
        Space::Normalized => Ok(v),
        Space::OutputCanvas => normalize(v, output.canvas),
        Space::Output => {
            if let Some(crop) = output.crop {
                v[0] += crop[1] as f64;
                v[2] += crop[1] as f64;
                v[1] += crop[0] as f64;
                v[3] += crop[0] as f64;
            }
            normalize(v, output.canvas)
        }
        _ => Err(invalid(
            "target coordinates require normalized, output or output_canvas space",
        )),
    }
}

fn source_coords(
    mut v: [f64; 4],
    space: Space,
    binding: &ReferenceBinding,
    geometry: Option<&references::ReferenceGeometry>,
) -> Result<[f64; 4]> {
    match space {
        Space::Normalized => return Ok(v),
        Space::Original => v = orient_box(v, binding.original_size, binding.orientation)?,
        Space::Oriented => {}
        Space::Fitted | Space::ReferenceCanvas => {
            let g = geometry
                .ok_or_else(|| invalid("source transform needs actual reference geometry"))?;
            if space == Space::ReferenceCanvas {
                let dx = g.source_region[0] as f64 - g.canvas_region[0] as f64;
                let dy = g.source_region[1] as f64 - g.canvas_region[1] as f64;
                v[0] += dy;
                v[2] += dy;
                v[1] += dx;
                v[3] += dx;
            }
            // Invert the source's resize/crop/pad. The caption addresses the original oriented
            // reference, not negative-l AE token coordinates or the prepared canvas.
            return normalize(v, g.fitted_size);
        }
        _ => {
            return Err(invalid(
                "source coordinates require a reference coordinate space",
            ));
        }
    }
    normalize(v, binding.oriented_size())
}

/// Transform all four rectangle corners, using continuous pixel EDGE coordinates (no -1).
/// This is authoring correspondence only; pixel EXIF preprocessing is owned by the decoder.
pub fn orient_box(v: [f64; 4], size: (u32, u32), orientation: u8) -> Result<[f64; 4]> {
    if !(1..=8).contains(&orientation)
        || size.0 == 0
        || size.1 == 0
        || v.iter().any(|n| !n.is_finite())
    {
        return Err(invalid("invalid original image orientation/coordinates"));
    }
    let (w, h) = (size.0 as f64, size.1 as f64);
    let mut out = [
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    for (x, y) in [(v[1], v[0]), (v[3], v[0]), (v[1], v[2]), (v[3], v[2])] {
        let (x, y) = match orientation {
            1 => (x, y),
            2 => (w - x, y),
            3 => (w - x, h - y),
            4 => (x, h - y),
            5 => (y, x),
            6 => (h - y, x),
            7 => (h - y, w - x),
            8 => (y, w - x),
            _ => unreachable!(),
        };
        out[0] = out[0].min(y);
        out[1] = out[1].min(x);
        out[2] = out[2].max(y);
        out[3] = out[3].max(x);
    }
    Ok(out)
}
