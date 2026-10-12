//! Opt-in MiniMax-H3 learned sparse attention. An absent or disabled request preserves dense execution.

/// A user budget is either a percentage skipped or a positive integer number of key tiles.
/// Internal fractional tile allocation belongs to the selector, never to user input.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum VedaBudget {
    SparsityPercent(f64),
    KeyTiles(u32),
}
impl Default for VedaBudget {
    fn default() -> Self {
        Self::SparsityPercent(90.0)
    }
}
impl VedaBudget {
    pub fn validate(self) -> Result<(), String> {
        match self {
            Self::SparsityPercent(p) if p.is_finite() && (0.0..100.0).contains(&p) => Ok(()),
            Self::KeyTiles(n) if n > 0 => Ok(()),
            Self::SparsityPercent(_) => {
                Err("Veda sparsity percentage must be finite and in [0, 100)".into())
            }
            Self::KeyTiles(_) => Err("Veda key-tile count must be a positive integer".into()),
        }
    }
}
impl std::str::FromStr for VedaBudget {
    type Err = String;
    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let value = text.trim();
        let budget = if let Some(percent) = value.strip_suffix('%') {
            Self::SparsityPercent(percent.trim().parse().map_err(|_| {
                "Veda sparsity must be a percentage or positive integer key-tile count".to_owned()
            })?)
        } else {
            if value.is_empty() || !value.bytes().all(|c| c.is_ascii_digit()) {
                return Err(
                    "Veda sparsity must be a percentage or positive integer key-tile count".into(),
                );
            }
            Self::KeyTiles(
                value
                    .parse()
                    .map_err(|_| "Veda key-tile count exceeds u32".to_owned())?,
            )
        };
        budget.validate()?;
        Ok(budget)
    }
}

/// Request controls use the frozen Veda defaults. Providers report what actually executed.
#[derive(Clone, Debug, PartialEq)]
pub struct VedaRequest {
    pub enabled: bool,
    pub generated: VedaBudget,
    pub reference: VedaBudget,
    /// Zero-based DiT blocks to run with full attention.
    pub full_attention_layers: Vec<usize>,
    /// Zero-based actual sampling evaluations to run with full attention.
    pub full_attention_steps: Vec<usize>,
}
impl Default for VedaRequest {
    fn default() -> Self {
        Self {
            enabled: false,
            generated: Default::default(),
            reference: Default::default(),
            full_attention_layers: vec![],
            full_attention_steps: vec![],
        }
    }
}
impl VedaRequest {
    /// Validate even disabled controls so malformed persisted settings cannot hide until enabled.
    pub fn validate(&self, layers: usize, steps: usize) -> Result<(), String> {
        self.generated
            .validate()
            .map_err(|e| format!("generated: {e}"))?;
        self.reference
            .validate()
            .map_err(|e| format!("reference: {e}"))?;
        for (name, indices, domain) in [
            ("full_attention_layers", &self.full_attention_layers, layers),
            ("full_attention_steps", &self.full_attention_steps, steps),
        ] {
            if let Some(index) = indices.iter().find(|&&i| i >= domain) {
                return Err(format!(
                    "Veda {name}: zero-based index {index} outside 0..{domain}"
                ));
            }
        }
        Ok(())
    }
    pub fn full_attention(&self, layer: usize, step: usize) -> bool {
        !self.enabled
            || self.full_attention_layers.contains(&layer)
            || self.full_attention_steps.contains(&step)
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct VedaExecutionReport {
    pub requested: bool,
    pub kernel: Option<String>,
    pub predictor: Option<String>,
    pub plan: Option<String>,
    pub exact_plan: bool,
    pub predictor_load_attempts: u64,
    pub predictor_score_calls: u64,
    pub sparse_calls: u64,
    pub dense_calls: u64,
    pub selected_token_pairs: u64,
    pub dense_token_pairs: u64,
    /// Maximum accounted workspace of any executed attention call, excluding the caller's input QKV.
    pub workspace_bytes: u64,
    /// Engine-observed reason for dense fallback; never inferred from requested settings.
    pub fallback_reason: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn frozen_controls_and_independent_budgets() {
        let mut request = VedaRequest {
            enabled: true,
            ..Default::default()
        };
        assert_eq!(request.generated, VedaBudget::SparsityPercent(90.0));
        assert_eq!(request.reference, VedaBudget::SparsityPercent(90.0));
        request.generated = "12".parse().unwrap();
        request.reference = "0%".parse().unwrap();
        assert_eq!(request.generated, VedaBudget::KeyTiles(12));
        assert_eq!(request.reference, VedaBudget::SparsityPercent(0.0));
        for steps in [4, 8, 50] {
            request.full_attention_layers = vec![0, 49];
            request.full_attention_steps = vec![0, steps - 1];
            request.validate(50, steps).unwrap();
            assert!(request.full_attention(49, 1));
            assert!(request.full_attention(1, steps - 1));
            assert!(!request.full_attention(1, 1));
            assert!(request.validate(49, steps).is_err());
            assert!(request.validate(50, steps - 1).is_err());
        }
    }
    #[test]
    fn malformed_controls_are_refused() {
        for input in [
            "",
            "0",
            "-1",
            "1.5",
            "32.0",
            "NaN%",
            "inf%",
            "100%",
            "-0.01%",
            "4294967296",
            "+1",
        ] {
            assert!(input.parse::<VedaBudget>().is_err(), "{input}");
        }
        for bad in [
            VedaBudget::KeyTiles(0),
            VedaBudget::SparsityPercent(f64::NAN),
            VedaBudget::SparsityPercent(100.0),
        ] {
            let r = VedaRequest {
                reference: bad,
                ..Default::default()
            };
            assert!(r.validate(50, 50).is_err());
        }
    }
}
