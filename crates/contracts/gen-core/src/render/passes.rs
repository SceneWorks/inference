//! Ordered multi-pass image plans whose later passes consume an earlier pass's output.
//!
//! Hires.fix renders a base pass, stretches its image to the target size and renders a second,
//! noise-start pass that conditions on it. Every pass is an ordinary [`RenderPlan`] compiled by
//! the provider's own pre-load compiler, so pass geometry, steps and refusal rules are checked
//! before load exactly as for a single-pass request. A produced reference is bound by its declared
//! geometry and strength instead of a pixel digest, because its bytes exist only after the earlier
//! pass runs. The caller executes the passes in order, each through
//! [`crate::Generator::generate_planned`] with `plan.passes()[i]`.

use super::{RenderPlan, Serialize};
use crate::{Conditioning, Error, GenerationRequest, Image, Result};

/// Largest produced-reference placeholder the compiler allocates (4096² RGB). Providers refuse
/// larger sizes themselves; this only bounds the stand-in buffer used for compilation.
const MAX_PRODUCED_REFERENCE_PIXELS: u64 = 4096 * 4096;

/// A reference image that pass `from_pass` produces, resized by the caller to exactly
/// `width × height` RGB before the consuming pass runs.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct ProducedReference {
    pub from_pass: usize,
    pub width: u32,
    pub height: u32,
    pub strength: Option<f32>,
}

/// One pass to compile. A pass that declares `produced_reference` must carry no conditioning of
/// its own: the compiler supplies the declared reference.
#[derive(Clone, Copy, Debug)]
pub struct PlannedPassRequest<'a> {
    pub request: &'a GenerationRequest,
    pub produced_reference: Option<ProducedReference>,
}

/// Ordered passes compiled before load for one provider and one load contract.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MultiPassRenderPlan {
    passes: Vec<RenderPlan>,
}

impl MultiPassRenderPlan {
    /// Compile every pass with `compile` (the provider's single-pass pre-load compiler) and
    /// validate the whole plan. `Ok(None)` means the provider has no pre-load plan for a pass.
    pub fn compile(
        passes: &[PlannedPassRequest<'_>],
        mut compile: impl FnMut(&GenerationRequest) -> Result<Option<RenderPlan>>,
    ) -> Result<Option<Self>> {
        if passes.len() < 2 {
            return Err(Error::Unsupported(
                "a multi-pass plan needs at least two passes".into(),
            ));
        }
        let mut compiled = Vec::with_capacity(passes.len());
        for (index, pass) in passes.iter().enumerate() {
            let plan = match pass.produced_reference {
                None => compile(pass.request)?,
                Some(produced) => {
                    if produced.from_pass >= index {
                        return Err(Error::Unsupported(format!(
                            "pass {index} can only consume an earlier pass's output, not pass {}",
                            produced.from_pass
                        )));
                    }
                    if !pass.request.conditioning.is_empty() {
                        return Err(Error::Unsupported(format!(
                            "pass {index} declares a produced reference and must carry no other \
                             conditioning"
                        )));
                    }
                    let request = with_placeholder_reference(pass.request, produced)?;
                    compile(&request)?
                        .map(|plan| plan.into_produced_reference(produced.from_pass))
                        .transpose()?
                }
            };
            let Some(mut plan) = plan else {
                if index == 0 {
                    return Ok(None);
                }
                return Err(Error::Unsupported(format!(
                    "pass {index} has no pre-load plan while pass 0 does"
                )));
            };
            plan.pass_index = index;
            compiled.push(plan);
        }
        Self::new(compiled).map(Some)
    }

    fn new(passes: Vec<RenderPlan>) -> Result<Self> {
        let first = &passes[0];
        for (index, pass) in passes.iter().enumerate() {
            if pass.provider != first.provider
                || pass.backend != first.backend
                || pass.preparation() != first.preparation()
            {
                return Err(Error::Unsupported(format!(
                    "pass {index} belongs to a different provider or load contract"
                )));
            }
            // Each pass renders one image that the next pass consumes; a prompt batch would
            // leave the produced input ambiguous.
            if pass.image_count() != 1 {
                return Err(Error::Unsupported(format!(
                    "pass {index} of a multi-pass plan must render exactly one image"
                )));
            }
            if let Some(produced) = pass.produced_reference() {
                if produced.from_pass >= index {
                    return Err(Error::Unsupported(format!(
                        "pass {index} consumes the output of a later pass"
                    )));
                }
            }
        }
        Ok(Self { passes })
    }

    /// The passes in execution order.
    pub fn passes(&self) -> &[RenderPlan] {
        &self.passes
    }

    /// Executed denoise steps summed over every pass: the progress total of the whole render.
    pub fn total_executed_steps(&self) -> usize {
        self.passes.iter().map(RenderPlan::executed_steps).sum()
    }

    /// Every pass shares one load contract; check it against the loaded provider once.
    pub fn check_load(&self, loaded: &super::RenderLoadBinding) -> Result<()> {
        self.passes
            .iter()
            .try_for_each(|pass| pass.check_load(loaded))
    }
}

fn with_placeholder_reference(
    request: &GenerationRequest,
    produced: ProducedReference,
) -> Result<GenerationRequest> {
    let pixels = u64::from(produced.width) * u64::from(produced.height);
    if pixels == 0 || pixels > MAX_PRODUCED_REFERENCE_PIXELS {
        return Err(Error::Unsupported(format!(
            "produced reference {}x{} is outside 1..={MAX_PRODUCED_REFERENCE_PIXELS} pixels",
            produced.width, produced.height
        )));
    }
    let mut request = request.clone();
    request.conditioning = vec![Conditioning::Reference {
        image: Image {
            width: produced.width,
            height: produced.height,
            pixels: vec![0; pixels as usize * 3],
        },
        strength: produced.strength,
    }];
    Ok(request)
}
