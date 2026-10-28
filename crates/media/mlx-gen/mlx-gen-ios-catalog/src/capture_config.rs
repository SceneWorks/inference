//! Typed caller-provisioned adapter configuration for the device capture harness. This transports
//! actual LoadSpec fields; no caller-claimed fit/class/derivation metadata is accepted.
use std::path::{Component, Path};

use mlx_gen::gen_core::{AdapterKind, AdapterSpec, Error, MoeExpert, Result};
use serde::Deserialize;

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Kind {
    Lora,
    Lokr,
}
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Expert {
    High,
    Low,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Adapter {
    path: String,
    scale: f32,
    kind: Kind,
    pass_scales: Option<Vec<f32>>,
    moe_expert: Option<Expert>,
}

/// Parse `adapters.json`: a typed list of the real AdapterSpec fields. Paths are confined relative
/// FILE paths inside the caller-provisioned slot; lexical binding is preserved (no canonicalize).
/// Header normalization/LoHa detection and legal forms remain the actual provider's responsibility.
pub fn capture_adapters(bytes: &[u8], slot: &Path) -> Result<Vec<AdapterSpec>> {
    let adapters: Vec<Adapter> = serde_json::from_slice(bytes)
        .map_err(|e| Error::Msg(format!("invalid typed adapter manifest: {e}")))?;
    adapters
        .into_iter()
        .map(|adapter| {
            let relative = Path::new(&adapter.path);
            if adapter.path.is_empty()
                || relative.is_absolute()
                || relative
                    .components()
                    .any(|part| !matches!(part, Component::Normal(_)))
            {
                return Err(Error::Msg(
                    "capture adapter paths must be confined relative files".into(),
                ));
            }
            if !adapter.scale.is_finite()
                || adapter
                    .pass_scales
                    .as_ref()
                    .is_some_and(|scales| scales.iter().any(|scale| !scale.is_finite()))
            {
                return Err(Error::Msg("capture adapter scales must be finite".into()));
            }
            let path = slot.join(relative);
            if !path.is_file() {
                return Err(Error::Msg(format!(
                    "missing provisioned adapter {}; not admission/refusal evidence",
                    path.display()
                )));
            }
            Ok(AdapterSpec {
                path,
                scale: adapter.scale,
                kind: match adapter.kind {
                    Kind::Lora => AdapterKind::Lora,
                    Kind::Lokr => AdapterKind::Lokr,
                },
                pass_scales: adapter.pass_scales,
                moe_expert: adapter.moe_expert.map(|expert| match expert {
                    Expert::High => MoeExpert::High,
                    Expert::Low => MoeExpert::Low,
                }),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn typed_capture_adapter_fields_are_preserved_and_unclaimed() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("adapter.safetensors"),
            b"header parsed by real provider later",
        )
        .unwrap();
        let parsed = capture_adapters(br#"[{"path":"adapter.safetensors","scale":0.5,"kind":"lokr","pass_scales":[0.25,0.75],"moe_expert":"high"}]"#, dir.path()).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].path, dir.path().join("adapter.safetensors"));
        assert_eq!(parsed[0].scale, 0.5);
        assert_eq!(parsed[0].kind, AdapterKind::Lokr);
        assert_eq!(parsed[0].pass_scales, Some(vec![0.25, 0.75]));
        assert_eq!(parsed[0].moe_expert, Some(MoeExpert::High));
        for invalid in [
            br#"[{"path":"../adapter.safetensors","scale":1.0,"kind":"lora"}]"#.as_slice(),
            br#"[{"path":"adapter.safetensors","scale":1.0,"kind":"lora","class":"resident-fit"}]"#
                .as_slice(),
            br#"[{"path":"missing.safetensors","scale":1.0,"kind":"lora"}]"#.as_slice(),
        ] {
            assert!(capture_adapters(invalid, dir.path()).is_err());
        }
    }
}
