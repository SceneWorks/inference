//! MOSS's bias-corrected Adam/AdamW policy. Other MLX families retain their
//! established mlx-rs optimizer convention; Rose and Prodigy are unchanged.
use super::*;
use mlx_gen::train::lora::{apply_weight_noise, clip_and_noise_grads};
use mlx_rs::optimizers::{Optimizer, OptimizerState};
use std::collections::{BTreeSet, HashMap};

const POLICY: &str = "moss-bias-corrected-adamw-v1";
const POLICY_KEY: &str = "moss_optimizer_policy";
const CLOCK_KEY: &str = "moss_optimizer_updates";

pub(crate) struct MossOptimizer {
    inner: mlx_gen::TrainOptimizer,
    adam_updates: Option<u32>,
}
impl MossOptimizer {
    pub(crate) fn is_supported(name: &str) -> bool {
        mlx_gen::TrainOptimizer::is_supported(name)
    }
    pub(crate) fn from_config(name: &str, lr: f32, weight_decay: f32) -> Result<Self> {
        let inner = mlx_gen::TrainOptimizer::from_config(name, lr, weight_decay)?;
        let adam_updates = matches!(inner, mlx_gen::TrainOptimizer::AdamW { .. }).then_some(0);
        Ok(Self {
            inner,
            adam_updates,
        })
    }
    pub(crate) fn kind_tag(&self) -> &'static str {
        self.inner.kind_tag()
    }
    pub(crate) fn set_lr_scaled(&mut self, multiplier: f32) {
        self.inner.set_lr_scaled(multiplier);
    }
    pub(crate) fn update(
        &mut self,
        params: &mut LoraParams,
        averaged: &LoraParams,
        cfg: &TrainingConfig,
        update: u32,
        seed: u64,
    ) -> Result<()> {
        let Some(completed) = self.adam_updates else {
            return mlx_gen::train::lora::adapter_optimizer_update(
                &mut self.inner,
                params,
                averaged,
                cfg,
                update,
                seed,
            );
        };
        if completed != update
            || params.keys().collect::<BTreeSet<_>>() != averaged.keys().collect()
        {
            return Err(Error::Unsupported(
                "MOSS AdamW update clock/parameter population differs".into(),
            ));
        }
        let next = completed
            .checked_add(1)
            .ok_or_else(|| Error::Unsupported("MOSS AdamW update clock overflow".into()))?;
        let grads = clip_and_noise_grads(averaged, cfg, update, seed)?;
        let mlx_gen::TrainOptimizer::AdamW { opt, .. } = &mut self.inner else {
            unreachable!()
        };
        let m_scale = Array::from_f32((1. / (1. - 0.9f64.powf(f64::from(next)))) as f32);
        let v_scale = Array::from_f32((1. / (1. - 0.999f64.powf(f64::from(next)))) as f32);
        // Decay uses the original scheduled LR; epsilon follows sqrt(v_hat).
        // Rescaling LR alone would implement neither of these AdamW contracts.
        let decay = Array::from_f32(1.).subtract(opt.lr.multiply(&opt.weight_decay)?)?;
        let lr = opt.lr.clone();
        let mut keys: Vec<_> = params.keys().cloned().collect();
        keys.sort();
        for key in keys {
            let gradient = &grads[&key];
            let parameter = &params[&key];
            let (m, v) = opt
                .state_mut()
                .entry(key.clone())
                .or_insert_with(|| (Array::from_f32(0.), Array::from_f32(0.)));
            let next_m = m
                .multiply(Array::from_f32(0.9))?
                .add(gradient.multiply(Array::from_f32(0.1))?)?;
            let next_v = v
                .multiply(Array::from_f32(0.999))?
                .add(gradient.square()?.multiply(Array::from_f32(0.001))?)?;
            let corrected = next_m.multiply(&m_scale)?.divide(
                next_v
                    .multiply(&v_scale)?
                    .sqrt()?
                    .add(Array::from_f32(1e-8))?,
            )?;
            let value = parameter
                .multiply(&decay)?
                .subtract(corrected.multiply(&lr)?)?;
            *m = next_m;
            *v = next_v;
            params.insert(key, value);
        }
        eval(params.values())?;
        apply_weight_noise(params, cfg.weight_noise_sigma, seed, update)?;
        self.adam_updates = Some(next);
        Ok(())
    }
    pub(crate) fn save_state(&self, path: impl AsRef<Path>) -> Result<()> {
        if let mlx_gen::TrainOptimizer::AdamW { opt, .. } = &self.inner {
            let metadata = HashMap::from([
                (POLICY_KEY.to_owned(), POLICY.to_owned()),
                (
                    CLOCK_KEY.to_owned(),
                    self.adam_updates.expect("AdamW clock").to_string(),
                ),
            ]);
            Array::save_safetensors(opt.state().flatten(), Some(&metadata), path)?;
            Ok(())
        } else {
            self.inner.save_state(path)
        }
    }
    pub(crate) fn load_state_checked(
        &mut self,
        path: impl AsRef<Path>,
        params: &LoraParams,
        updates: u32,
    ) -> Result<()> {
        let path = path.as_ref();
        if self.adam_updates.is_some() {
            let (_, metadata) = mlx_gen::weights::load_safetensors_on_cpu(path)?;
            validate_adam_metadata(&metadata, updates)?;
        }
        self.inner.load_state_checked(path, params, updates)?;
        if self.adam_updates.is_some() {
            self.adam_updates = Some(updates);
        }
        Ok(())
    }
}
fn validate_adam_metadata(metadata: &HashMap<String, String>, updates: u32) -> Result<()> {
    if metadata.get(POLICY_KEY).map(String::as_str) != Some(POLICY)
        || metadata.get(CLOCK_KEY) != Some(&updates.to_string())
    {
        return Err(Error::Unsupported(
            "MOSS AdamW resume policy/clock differs; uncorrected checkpoints cannot be resumed"
                .into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod metadata_tests {
    use super::*;
    #[test]
    fn adam_resume_policy_and_optimizer_clock_are_explicit() {
        let valid = HashMap::from([
            (POLICY_KEY.to_owned(), POLICY.to_owned()),
            (CLOCK_KEY.to_owned(), "2".to_owned()),
        ]);
        validate_adam_metadata(&valid, 2).unwrap();
        for metadata in [
            HashMap::new(),
            HashMap::from([(POLICY_KEY.to_owned(), POLICY.to_owned())]),
            HashMap::from([
                (POLICY_KEY.to_owned(), "legacy".to_owned()),
                (CLOCK_KEY.to_owned(), "2".to_owned()),
            ]),
            HashMap::from([
                (POLICY_KEY.to_owned(), POLICY.to_owned()),
                (CLOCK_KEY.to_owned(), "3".to_owned()),
            ]),
        ] {
            assert!(validate_adam_metadata(&metadata, 2).is_err());
        }
    }
}
