//! Weights-free campaign protocol. One selected case per fresh process; no implicit fallbacks.
use std::collections::BTreeSet;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Surface {
    pub provider: String,
    pub routes: Vec<String>,
    pub modes: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Case {
    pub provider: String,
    pub route: String,
    pub tier: String,
    pub edge: u32,
    pub mode: String,
    pub rung: u8,
}
impl Case {
    pub fn key(&self) -> String {
        format!(
            "{}/{}/{}/{}/{}/{}",
            self.provider, self.route, self.tier, self.edge, self.mode, self.rung
        )
    }
    pub fn parse(value: &str) -> Result<Self, String> {
        let p: Vec<_> = value.split('/').collect();
        if p.len() != 6
            || p.iter().any(|p| {
                p.is_empty()
                    || !p
                        .chars()
                        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
            })
        {
            return Err("case must be provider/route/{bf16,q8,q4}/edge/mode/rung".into());
        }
        let edge = p[3].parse::<u32>().map_err(|_| "invalid edge")?;
        let rung = p[5].parse::<u8>().map_err(|_| "invalid rung")?;
        if !["bf16", "q8", "q4"].contains(&p[2])
            || ![256, 512, 768, 1024, 1280].contains(&edge)
            || rung > 4
        {
            return Err("tier, geometry or rung outside finite campaign domain".into());
        }
        Ok(Self {
            provider: p[0].into(),
            route: p[1].into(),
            tier: p[2].into(),
            edge,
            mode: p[4].into(),
            rung,
        })
    }
}

/// Include every registered mode, even if its adapter/evidence is not yet available. A new
/// provider is visible automatically; admission, never enumeration, decides whether it can run.
pub fn enumerate(surfaces: &[Surface]) -> Vec<Case> {
    let mut cases = Vec::new();
    let mut seen = BTreeSet::new();
    for s in surfaces {
        for route in &s.routes {
            for tier in ["q4", "q8", "bf16"] {
                let edges: &[u32] = if s.provider.contains("supra") {
                    &[256]
                } else {
                    &[512, 768, 1024]
                };
                for &edge in edges {
                    for mode in &s.modes {
                        for rung in 0..=4 {
                            let c = Case {
                                provider: s.provider.clone(),
                                route: route.clone(),
                                tier: tier.into(),
                                edge,
                                mode: mode.clone(),
                                rung,
                            };
                            if seen.insert(c.key()) {
                                cases.push(c);
                            }
                        }
                    }
                }
            }
        }
    }
    cases
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Successful,
    AdmittedRefused,
    DiagnosticSuccess,
    DiagnosticFailure,
    Skipped,
}
impl Status {
    pub fn label(self) -> &'static str {
        match self {
            Self::Successful => "successful",
            Self::AdmittedRefused => "admitted_refused",
            Self::DiagnosticSuccess => "diagnostic_success",
            Self::DiagnosticFailure => "diagnostic_failure",
            Self::Skipped => "skipped",
        }
    }
    pub fn production_success(self) -> bool {
        self == Self::Successful
    }
}

/// The adapter MUST complete admission before entering `execute`. Tests use the identical
/// orchestration with a fake provider, rather than testing a second implementation of the gate.
#[derive(Debug)]
pub enum AdmissionError {
    Refused(String),
    Failure(String),
}
impl From<String> for AdmissionError {
    fn from(e: String) -> Self {
        Self::Failure(e)
    }
}
impl From<&str> for AdmissionError {
    fn from(e: &str) -> Self {
        Self::Failure(e.into())
    }
}

pub trait Provider {
    fn admit(&mut self, case: &Case, diagnostic: bool) -> Result<(), AdmissionError>;
    fn execute(&mut self, case: &Case) -> Result<(), String>;
}
pub fn run(provider: &mut impl Provider, case: &Case, diagnostic: bool) -> (Status, String) {
    if let Err(error) = provider.admit(case, diagnostic) {
        return match error {
            AdmissionError::Refused(reason) => (
                if diagnostic {
                    Status::DiagnosticFailure
                } else {
                    Status::AdmittedRefused
                },
                reason,
            ),
            AdmissionError::Failure(reason) => (Status::DiagnosticFailure, reason),
        };
    }
    match provider.execute(case) {
        Ok(()) => (
            if diagnostic {
                Status::DiagnosticSuccess
            } else {
                Status::Successful
            },
            "completed".into(),
        ),
        Err(e) => (Status::DiagnosticFailure, e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn surface() -> Surface {
        Surface {
            provider: "new_provider".into(),
            routes: vec!["new_provider".into()],
            modes: vec!["text".into(), "reference".into(), "mask".into()],
        }
    }
    #[test]
    fn public_case_parse_roundtrip_and_finite_budget() {
        let c = Case::parse("iris_3b/iris_3b/q8/768/reference/4").unwrap();
        assert_eq!(c.edge, 768);
        assert_eq!(c.rung, 4);
        assert_eq!(c.tier, "q8");
        assert_eq!(Case::parse(&c.key()).unwrap(), c);
        for s in [
            "p/p/q4/0/text/1",
            "p/p/q4/999999999/text/1",
            "p/p/q4/512/text/5",
            "p/../q4/512/text/0",
            "p/p/q5/512/text/0",
            "p/p/q4/512/text/0/extra",
        ] {
            assert!(Case::parse(s).is_err(), "{s}");
        }
    }
    #[test]
    fn catalog_drives_all_modes_tiers_geometries_and_rungs() {
        let s = surface();
        let cases = enumerate(&[s.clone(), s]);
        let actual: BTreeSet<_> = cases.iter().map(Case::key).collect();
        assert_eq!(actual.len(), cases.len());
        for tier in ["q4", "q8", "bf16"] {
            for edge in [512, 768, 1024] {
                for mode in ["text", "reference", "mask"] {
                    for rung in 0..=4 {
                        assert!(actual.contains(&format!(
                            "new_provider/new_provider/{tier}/{edge}/{mode}/{rung}"
                        )));
                    }
                }
            }
        }
        let supra = Surface {
            provider: "supra2_img".into(),
            ..surface()
        };
        assert!(enumerate(&[supra]).iter().all(|c| c.edge == 256));
    }
    struct Fake {
        admitted: bool,
        fail: bool,
        loads: usize,
    }
    impl Provider for Fake {
        fn admit(&mut self, _: &Case, _: bool) -> Result<(), AdmissionError> {
            if self.admitted {
                Ok(())
            } else {
                Err(AdmissionError::Refused("no bound".into()))
            }
        }
        fn execute(&mut self, _: &Case) -> Result<(), String> {
            self.loads += 1;
            if self.fail {
                Err("bad output".into())
            } else {
                Ok(())
            }
        }
    }
    #[test]
    fn next_up_refuses_before_weight_load_and_no_false_diagnostic_pass() {
        let c = enumerate(&[surface()]).remove(0);
        let mut p = Fake {
            admitted: false,
            fail: false,
            loads: 0,
        };
        assert_eq!(run(&mut p, &c, false).0, Status::AdmittedRefused);
        assert_eq!(p.loads, 0);
        assert_eq!(run(&mut p, &c, true).0, Status::DiagnosticFailure);
        assert_eq!(p.loads, 0);
        p.admitted = true;
        assert_eq!(run(&mut p, &c, false).0, Status::Successful);
        assert_eq!(p.loads, 1);
        let diagnostic = run(&mut p, &c, true).0;
        assert_eq!(diagnostic, Status::DiagnosticSuccess);
        assert!(!diagnostic.production_success());
        p.fail = true;
        assert_eq!(run(&mut p, &c, false).0, Status::DiagnosticFailure);
        assert!(!Status::Skipped.production_success());
    }
}
