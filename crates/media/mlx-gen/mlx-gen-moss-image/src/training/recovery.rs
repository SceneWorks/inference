//! Complete MOSS recovery bundles, keeping generic legacy trainer snapshots compatible.
use super::*;
use contract::recovery::{bundle, DataPlan, RunState};
use std::{collections::HashMap, rc::Rc};
fn finite(value: &Array) -> Result<()> {
    if !value.is_finite()?.all(None)?.item::<bool>() {
        return Err(Error::Unsupported(
            "MOSS resume contains a nonfinite tensor".into(),
        ));
    }
    Ok(())
}
pub(super) fn save_complete(
    req: &TrainingRequest,
    stem: &str,
    state: &RunState,
    opt: &TrainOptimizer,
    params: &LoraParams,
    pending: Option<&LoraParams>,
) -> Result<()> {
    bundle::publish(&req.output_dir, stem, state, |dir| {
        let mut tensors = Vec::new();
        for (name, value) in params {
            finite(value)?;
            tensors.push((format!("factor.{name}"), value));
            if state.accumulated_tokens > 0 {
                let gradient = pending.and_then(|p| p.get(name)).ok_or_else(|| {
                    gen_core::Error::Msg(format!("MOSS missing pending gradient {name}"))
                })?;
                finite(gradient)?;
                tensors.push((format!("pending.{name}"), gradient));
            }
        }
        let metadata = HashMap::from([("optimizer".to_owned(), opt.kind_tag().to_owned())]);
        Array::save_safetensors(tensors, Some(&metadata), dir.join("tensors.safetensors"))
            .map_err(Error::from)?;
        opt.save_state(dir.join("optimizer.safetensors"))?;
        Ok(())
    })?;
    Ok(())
}
pub(super) fn load_complete(
    req: &TrainingRequest,
    stem: &str,
    plan: &DataPlan,
    identity: &contract::recovery::RunIdentity,
    opt: &mut TrainOptimizer,
    expected: &LoraParams,
) -> Result<(RunState, LoraParams, Option<LoraParams>)> {
    let path = bundle::latest(&req.output_dir, stem)?;
    let state = bundle::read(&path)?;
    state.validate(req, plan, identity)?;
    let (tensors, metadata) =
        mlx_gen::weights::load_safetensors_on_cpu(path.join("tensors.safetensors"))?;
    mlx_rs::transforms::eval_pending_loads(tensors.values())?;
    if metadata.get("optimizer").map(String::as_str) != Some(opt.kind_tag()) {
        return Err(Error::Unsupported("MOSS optimizer kind differs".into()));
    }
    let mut factors = LoraParams::new();
    let mut pending = LoraParams::new();
    for (name, value) in tensors {
        finite(&value)?;
        if value.dtype() != Dtype::Float32 {
            return Err(Error::Unsupported("MOSS raw state must be FP32".into()));
        }
        let (name, target) = if let Some(name) = name.strip_prefix("factor.") {
            (name, &mut factors)
        } else if let Some(name) = name.strip_prefix("pending.") {
            (name, &mut pending)
        } else {
            return Err(Error::Unsupported("MOSS unknown resume tensor".into()));
        };
        let expected = expected
            .get(name)
            .ok_or_else(|| Error::Unsupported("MOSS resume target surface differs".into()))?;
        if expected.shape() != value.shape() {
            return Err(Error::Unsupported(
                "MOSS resume factor/gradient shape differs".into(),
            ));
        }
        target.insert(Rc::from(name), value);
    }
    if factors.len() != expected.len()
        || pending.len()
            != if state.accumulated_tokens > 0 {
                expected.len()
            } else {
                0
            }
    {
        return Err(Error::Unsupported(
            "MOSS incomplete factor/pending-gradient surface".into(),
        ));
    }
    opt.load_state_checked(path.join("optimizer.safetensors"), expected, state.update)?;
    let pending = if state.accumulated_tokens > 0 {
        Some(pending)
    } else {
        None
    };
    Ok((state, factors, pending))
}
