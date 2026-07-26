//! Adapter checkpoint naming plus full-state resume bundles. Intermediate PEFT checkpoints remain
//! user-loadable; their `.resume.safetensors` siblings hold raw factors, optimizer state, and the
//! schedule cursor so an interrupted candle run can continue exactly.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::{Device, Tensor};

use crate::gen_core::train::TrainingConfig;
use crate::train::lora::LoraSet;
use crate::train::optim::TrainOptimizer;
use crate::{CandleError, Result};

/// Strip a trailing `.safetensors` from an adapter file name to get the stem used for intermediate
/// checkpoints (`my_style.safetensors` → `my_style`).
pub fn file_stem(file_name: &str) -> &str {
    file_name.strip_suffix(".safetensors").unwrap_or(file_name)
}

/// `{stem}-step{step:06}.safetensors` — the intermediate-checkpoint file name at micro-step `step`.
pub fn checkpoint_filename(stem: &str, step: u32) -> String {
    format!("{stem}-step{step:06}.safetensors")
}

/// `{stem}-step{step:06}.resume.safetensors`.
pub fn resume_snapshot_filename(stem: &str, step: u32) -> String {
    format!("{stem}-step{step:06}.resume.safetensors")
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResumeMeta {
    pub step: u32,
    pub update_idx: u32,
}

/// Find the highest-step resume snapshot for this adapter stem.
pub fn find_latest_resume(dir: &Path, stem: &str) -> Option<(PathBuf, u32)> {
    let prefix = format!("{stem}-step");
    let mut best = None;
    for entry in std::fs::read_dir(dir).ok()?.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let Some(rest) = name.strip_prefix(&prefix) else {
            continue;
        };
        let Some(digits) = rest.strip_suffix(".resume.safetensors") else {
            continue;
        };
        if let Ok(step) = digits.parse::<u32>() {
            if best.as_ref().is_none_or(|(_, current)| step > *current) {
                best = Some((entry.path(), step));
            }
        }
    }
    best
}

/// Save raw factors and optimizer state in one safetensors file.
pub fn save_resume(
    dir: &Path,
    stem: &str,
    step: u32,
    update_idx: u32,
    opt: &TrainOptimizer,
    set: &LoraSet,
    cfg: &TrainingConfig,
) -> Result<PathBuf> {
    std::fs::create_dir_all(dir)
        .map_err(|e| CandleError::Msg(format!("resume: create {}: {e}", dir.display())))?;
    let mut tensors: HashMap<String, Tensor> = set
        .named_vars()
        .into_iter()
        .map(|(name, var)| (format!("factor.{name}"), var.as_tensor().clone()))
        .collect();
    let (optim_tensors, mut meta) = opt.snapshot()?;
    tensors.extend(optim_tensors);
    meta.insert("step".into(), step.to_string());
    meta.insert("update_idx".into(), update_idx.to_string());
    meta.insert("format".into(), "candle-gen-resume-v1".into());
    meta.insert("training_config".into(), training_fingerprint(cfg));
    let file_name = resume_snapshot_filename(stem, step);
    let path = dir.join(&file_name);
    // Same-directory temp + rename: `find_latest_resume` never observes a partially-written bundle.
    // Resume steps are append-only; a collision fails without replacing/corrupting the old snapshot
    // (portable across Windows and Unix, whose rename replacement semantics differ).
    if path.exists() {
        return Err(CandleError::Msg(format!(
            "resume: snapshot already exists at {}",
            path.display()
        )));
    }
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let temp = dir.join(format!(".{file_name}.{}.{nonce}.tmp", std::process::id()));
    if let Err(e) = safetensors::serialize_to_file(tensors.iter(), Some(meta), &temp) {
        std::fs::remove_file(&temp).ok();
        return Err(CandleError::Msg(format!(
            "resume: save {}: {e}",
            path.display()
        )));
    }
    if let Err(e) = std::fs::rename(&temp, &path) {
        std::fs::remove_file(&temp).ok();
        return Err(CandleError::Msg(format!(
            "resume: publish {}: {e}",
            path.display()
        )));
    }
    Ok(path)
}

/// Restore raw factors, optimizer state, and the schedule cursor.
pub fn load_resume(
    path: &Path,
    opt: &mut TrainOptimizer,
    set: &LoraSet,
    cfg: &TrainingConfig,
) -> Result<ResumeMeta> {
    let bytes = std::fs::read(path)
        .map_err(|e| CandleError::Msg(format!("resume: read {}: {e}", path.display())))?;
    let (_, header) = safetensors::SafeTensors::read_metadata(&bytes)
        .map_err(|e| CandleError::Msg(format!("resume: parse {}: {e}", path.display())))?;
    let meta = header.metadata().clone().unwrap_or_default();
    let parse = |key: &str| -> Result<u32> {
        meta.get(key)
            .ok_or_else(|| CandleError::Msg(format!("resume: missing metadata {key:?}")))?
            .parse()
            .map_err(|e| CandleError::Msg(format!("resume: invalid metadata {key:?}: {e}")))
    };
    if meta.get("format").map(String::as_str) != Some("candle-gen-resume-v1") {
        return Err(CandleError::Msg(
            "resume: unsupported snapshot format".into(),
        ));
    }
    let saved_config = meta
        .get("training_config")
        .ok_or_else(|| CandleError::Msg("resume: missing training_config metadata".into()))?;
    let requested_config = training_fingerprint(cfg);
    if saved_config != &requested_config {
        return Err(CandleError::Msg(format!(
            "resume: training configuration differs (saved {saved_config:?}, requested \
             {requested_config:?})"
        )));
    }
    let tensors = candle_core::safetensors::load(path, &Device::Cpu)?;
    let factors: HashMap<String, Tensor> = tensors
        .iter()
        .filter_map(|(key, value)| {
            key.strip_prefix("factor.")
                .map(|name| (name.to_string(), value.clone()))
        })
        .collect();
    set.restore_named(&factors)?;
    opt.restore(&tensors, &meta)?;
    Ok(ResumeMeta {
        step: parse("step")?,
        update_idx: parse("update_idx")?,
    })
}

fn training_fingerprint(cfg: &TrainingConfig) -> String {
    format!(
        "steps={};accum={};scheduler={:?};warmup={};rank={};alpha={};seed={};loss={};dtype={};\
         checkpoint={};timestep_type={};timestep_bias={}",
        cfg.steps,
        cfg.gradient_accumulation.max(1),
        cfg.lr_scheduler,
        cfg.lr_warmup_steps,
        cfg.rank,
        cfg.alpha,
        cfg.seed,
        cfg.loss_type,
        cfg.train_dtype,
        cfg.gradient_checkpointing,
        cfg.timestep_type,
        cfg.timestep_bias
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::lora::{build_lora_targets, LoraHost, LoraLinear};
    use candle_core::DType;
    use candle_nn::Linear;

    struct Host {
        linear: LoraLinear,
    }

    impl LoraHost for Host {
        fn visit_lora_mut(
            &mut self,
            f: &mut dyn FnMut(&mut LoraLinear) -> Result<()>,
        ) -> Result<()> {
            f(&mut self.linear)
        }
    }

    fn adapter(seed: u64) -> LoraSet {
        let device = Device::Cpu;
        let base = Linear::new(Tensor::zeros((4, 3), DType::F32, &device).unwrap(), None);
        let mut host = Host {
            linear: LoraLinear::from_linear(base, 3, 4, "blocks.0.attn1.to_q".into()),
        };
        build_lora_targets(&mut host, &["to_q".into()], 2, 2.0, seed, &device).unwrap()
    }

    fn write_bundle(
        path: &Path,
        set: &LoraSet,
        opt: &TrainOptimizer,
        cfg: &TrainingConfig,
        mutate: impl FnOnce(&mut HashMap<String, Tensor>),
    ) {
        let mut tensors: HashMap<String, Tensor> = set
            .named_vars()
            .into_iter()
            .map(|(name, var)| (format!("factor.{name}"), var.as_tensor().clone()))
            .collect();
        let (optim, mut meta) = opt.snapshot().unwrap();
        tensors.extend(optim);
        mutate(&mut tensors);
        meta.insert("step".into(), "4".into());
        meta.insert("update_idx".into(), "2".into());
        meta.insert("format".into(), "candle-gen-resume-v1".into());
        meta.insert("training_config".into(), training_fingerprint(cfg));
        safetensors::serialize_to_file(tensors.iter(), Some(meta), path).unwrap();
    }

    #[test]
    fn names_are_zero_padded_and_sortable() {
        assert_eq!(file_stem("my_style.safetensors"), "my_style");
        assert_eq!(file_stem("noext"), "noext");
        assert_eq!(
            checkpoint_filename("my_style", 500),
            "my_style-step000500.safetensors"
        );
        // Lexical sort matches numeric order.
        assert!(checkpoint_filename("s", 90) < checkpoint_filename("s", 100));
        assert_eq!(
            resume_snapshot_filename("my_style", 500),
            "my_style-step000500.resume.safetensors"
        );
    }

    #[test]
    fn resume_bundle_round_trips_and_rejects_factor_corruption() {
        let dir = std::env::temp_dir().join(format!(
            "candle_resume_bundle_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let source = adapter(7);
        let source_opt =
            TrainOptimizer::from_config("adamw", source.vars.clone(), 1e-3, 0.01).unwrap();
        let cfg = TrainingConfig {
            steps: 8,
            gradient_accumulation: 2,
            learning_rate: 1e-3,
            weight_decay: 0.01,
            ..Default::default()
        };
        let valid = save_resume(&dir, "adapter", 4, 2, &source_opt, &source, &cfg).unwrap();
        // A colliding step fails without replacing the valid old bundle and leaves no temp file.
        assert!(
            save_resume(&dir, "adapter", 4, 2, &source_opt, &source, &cfg)
                .unwrap_err()
                .to_string()
                .contains("already exists")
        );
        assert!(std::fs::read_dir(&dir)
            .unwrap()
            .flatten()
            .all(|entry| { !entry.file_name().to_string_lossy().ends_with(".tmp") }));

        let restored = adapter(99);
        let mut restored_opt =
            TrainOptimizer::from_config("adamw", restored.vars.clone(), 1e-3, 0.01).unwrap();
        let meta = load_resume(&valid, &mut restored_opt, &restored, &cfg).unwrap();
        assert_eq!(
            meta,
            ResumeMeta {
                step: 4,
                update_idx: 2
            }
        );
        for ((source_name, source_var), (restored_name, restored_var)) in
            source.named_vars().iter().zip(restored.named_vars().iter())
        {
            assert_eq!(source_name, restored_name);
            assert_eq!(
                source_var.as_tensor().to_vec2::<f32>().unwrap(),
                restored_var.as_tensor().to_vec2::<f32>().unwrap()
            );
        }

        let first = source.named_vars()[0].0.clone();
        for (name, mutate) in [
            (
                "missing",
                Box::new({
                    let first = first.clone();
                    move |map: &mut HashMap<String, Tensor>| {
                        map.remove(&format!("factor.{first}"));
                    }
                }) as Box<dyn FnOnce(&mut HashMap<String, Tensor>)>,
            ),
            (
                "extra",
                Box::new(|map: &mut HashMap<String, Tensor>| {
                    map.insert(
                        "factor.blocks.9.attn2.to_k.lora_A.weight".into(),
                        Tensor::zeros((2, 3), DType::F32, &Device::Cpu).unwrap(),
                    );
                }),
            ),
            (
                "shape",
                Box::new({
                    let first = first.clone();
                    move |map: &mut HashMap<String, Tensor>| {
                        map.insert(
                            format!("factor.{first}"),
                            Tensor::zeros((1, 1), DType::F32, &Device::Cpu).unwrap(),
                        );
                    }
                }),
            ),
        ] {
            let path = dir.join(format!("{name}.resume.safetensors"));
            write_bundle(&path, &source, &source_opt, &cfg, mutate);
            let target = adapter(99);
            let mut opt =
                TrainOptimizer::from_config("adamw", target.vars.clone(), 1e-3, 0.01).unwrap();
            let err = load_resume(&path, &mut opt, &target, &cfg)
                .unwrap_err()
                .to_string();
            assert!(
                err.contains("surface differs") || err.contains("shape"),
                "{name}: {err}"
            );
        }
        let mut changed = cfg.clone();
        changed.gradient_accumulation = 4;
        let target = adapter(99);
        let mut opt =
            TrainOptimizer::from_config("adamw", target.vars.clone(), 1e-3, 0.01).unwrap();
        assert!(load_resume(&valid, &mut opt, &target, &changed)
            .unwrap_err()
            .to_string()
            .contains("training configuration differs"));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
