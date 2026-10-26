//! MOSS-only complete-state persistence. The generic legacy trainer checkpoint format is unchanged.
use super::*;
use contract::recovery::{bundle, DataPlan, RunState};

pub(super) fn save_complete(
    req: &TrainingRequest,
    stem: &str,
    state: &RunState,
    opt: &TrainOptimizer,
    set: &LoraSet,
    window: &GradientWindow,
) -> Result<()> {
    bundle::publish(&req.output_dir, stem, state, |dir| {
        let mut tensors = HashMap::new();
        for (name, var) in set.named_vars() {
            tensors.insert(format!("factor.{name}"), var.as_tensor().clone());
            if state.accumulated_tokens > 0 {
                let gradient = window
                    .gradients
                    .as_ref()
                    .and_then(|g| g.get(var.as_tensor()))
                    .ok_or_else(|| {
                        gen_core::Error::Msg(format!(
                            "MOSS checkpoint missing pending gradient {name}"
                        ))
                    })?;
                crate::adapters::finite(gradient, "pending gradient")?;
                tensors.insert(format!("pending.{name}"), gradient.clone());
            }
        }
        safetensors::serialize_to_file(tensors.iter(), None, &dir.join("tensors.safetensors"))
            .map_err(|e| gen_core::Error::Msg(e.to_string()))?;
        let (optimizer, metadata) = opt.snapshot()?;
        safetensors::serialize_to_file(
            optimizer.iter(),
            Some(metadata),
            &dir.join("optimizer.safetensors"),
        )
        .map_err(|e| gen_core::Error::Msg(e.to_string()))?;
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
    set: &LoraSet,
) -> Result<(RunState, GradientWindow)> {
    let path = bundle::latest(&req.output_dir, stem)?;
    let state = bundle::read(&path)?;
    state.validate(req, plan, identity)?;
    let tensors =
        candle_gen::candle_core::safetensors::load(path.join("tensors.safetensors"), &Device::Cpu)?;
    let mut factors = HashMap::new();
    let mut pending = HashMap::new();
    for (name, value) in tensors {
        crate::adapters::finite(&value, "resume tensor")?;
        if value.dtype() != DType::F32 {
            return Err(CandleError::Msg(
                "MOSS raw resume tensors must be FP32".into(),
            ));
        }
        if let Some(name) = name.strip_prefix("factor.") {
            factors.insert(name.to_owned(), value);
        } else if let Some(name) = name.strip_prefix("pending.") {
            pending.insert(name.to_owned(), value);
        } else {
            return Err(CandleError::Msg(
                "MOSS resume contains an unknown tensor".into(),
            ));
        }
    }
    let named = set.named_vars();
    let expected = if state.accumulated_tokens > 0 {
        named.len()
    } else {
        0
    };
    if pending.len() != expected {
        return Err(CandleError::Msg(
            "MOSS pending gradient population differs".into(),
        ));
    }
    let mut gradients = GradStore::default();
    for (name, var) in &named {
        if state.accumulated_tokens > 0 {
            let value = pending
                .get(name)
                .ok_or_else(|| CandleError::Msg(format!("MOSS missing pending gradient {name}")))?;
            if value.dims() != var.dims() {
                return Err(CandleError::Msg(
                    "MOSS pending gradient shape differs".into(),
                ));
            }
            gradients.insert(var.as_tensor(), value.to_device(var.device())?);
        }
    }
    let optimizer_path = path.join("optimizer.safetensors");
    let bytes = std::fs::read(&optimizer_path).map_err(|e| CandleError::Msg(e.to_string()))?;
    let (_, header) = safetensors::SafeTensors::read_metadata(&bytes)
        .map_err(|e| CandleError::Msg(e.to_string()))?;
    let metadata = header
        .metadata()
        .clone()
        .ok_or_else(|| CandleError::Msg("MOSS missing optimizer metadata".into()))?;
    let optimizer = candle_gen::candle_core::safetensors::load(&optimizer_path, &Device::Cpu)?;
    for value in optimizer.values() {
        crate::adapters::finite(value, "optimizer state")?;
    }
    let mut expected_optimizer = std::collections::HashMap::<String, Vec<usize>>::new();
    match opt.kind_tag() {
        "adam" | "adamw" => {
            let saved = metadata
                .get("optim.step")
                .and_then(|s| s.parse::<u32>().ok())
                .ok_or_else(|| {
                    CandleError::Msg("MOSS missing/malformed optimizer update cursor".into())
                })?;
            if saved != state.update {
                return Err(CandleError::Msg(
                    "MOSS optimizer update cursor differs".into(),
                ));
            }
            for (index, variable) in set.vars.iter().enumerate() {
                for suffix in ["m", "v"] {
                    expected_optimizer
                        .insert(format!("optim.{index}.{suffix}"), variable.dims().to_vec());
                }
            }
        }
        "prodigy" => {
            for key in ["optim.d", "optim.d_max", "optim.d_numerator"] {
                let value = metadata
                    .get(key)
                    .and_then(|s| s.parse::<f64>().ok())
                    .ok_or_else(|| {
                        CandleError::Msg("MOSS missing Prodigy coupled scalar".into())
                    })?;
                if !value.is_finite() {
                    return Err(CandleError::Msg("MOSS nonfinite Prodigy scalar".into()));
                }
            }
            if state.update > 0 {
                for (index, variable) in set.vars.iter().enumerate() {
                    for suffix in ["exp_avg", "exp_avg_sq", "s", "p0"] {
                        expected_optimizer
                            .insert(format!("optim.{index}.{suffix}"), variable.dims().to_vec());
                    }
                }
            }
        }
        "rose" => (),
        _ => {
            return Err(CandleError::Msg(
                "MOSS unsupported optimizer state tag".into(),
            ))
        }
    }
    if optimizer.len() != expected_optimizer.len()
        || expected_optimizer.iter().any(|(key, shape)| {
            optimizer
                .get(key)
                .is_none_or(|t| t.dtype() != DType::F32 || t.dims() != shape.as_slice())
        })
    {
        return Err(CandleError::Msg(
            "MOSS incomplete/mismatched optimizer tensor surface".into(),
        ));
    }

    // Nothing is published to a live run before complete identity/tensor validation above.
    set.restore_named(&factors)?;
    opt.restore(&optimizer, &metadata)?;
    let window = GradientWindow {
        gradients: if state.accumulated_tokens > 0 {
            Some(gradients)
        } else {
            None
        },
        tokens: state.accumulated_tokens,
    };
    Ok((state, window))
}
