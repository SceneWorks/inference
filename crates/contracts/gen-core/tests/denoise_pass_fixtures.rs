//! Cross-language fixture conformance for `advanced.denoisePasses` (sc-20415, epic 20414).
//!
//! Every `.json` under `tests/fixtures/denoise_passes/` is read here **and** by the SceneWorks-side
//! PR of the same story. The two repositories must agree on: what decodes, what the canonical
//! re-encoding is, what a request resolves to (per-pass seeds included), and which pass index +
//! field a malformed request is rejected on. Keeping one copy of the fixtures — rather than a
//! paraphrase on each side — is the point; see the fixture directory's `README.md`.

use std::path::{Path, PathBuf};

use gen_core::denoise_passes::{
    denoise_passes_from_advanced_json, denoise_passes_to_json, json_equal_within,
    validate_denoise_passes, DenoiseDefaults, DenoisePass, DenoisePassContext, DenoisePassError,
    ResolvedDenoisePlan,
};
use gen_core::{Capabilities, DenoisePassSurface, Error, GenerationPhase, GenerationRequest};
use serde_json::Value;

/// The plan comparison tolerance. The contract's floats are `f32` and the fixtures are authored in
/// decimal, so an exact bit comparison would be asserting the decimal parser rather than the
/// contract. Every value in play is far coarser than this.
const PLAN_TOLERANCE: f64 = 1e-6;

fn fixture_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/denoise_passes")
}

fn read_json(path: &Path) -> Value {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|err| panic!("parse {}: {err}", path.display()))
}

fn json_files(dir: &Path) -> Vec<PathBuf> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|err| panic!("read_dir {}: {err}", dir.display()))
        .map(|entry| entry.expect("dir entry").path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "json"))
        .collect();
    paths.sort();
    assert!(!paths.is_empty(), "no fixtures under {}", dir.display());
    paths
}

fn stem(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .expect("utf-8 file name")
        .trim_end_matches(".json")
        .trim_end_matches(".plan")
        .to_string()
}

/// The fixture's declared model defaults — the last rung of the resolution ladder.
fn defaults_of(fixture: &Value) -> DenoiseDefaults {
    let defaults = fixture
        .get("modelDefaults")
        .expect("fixture declares `modelDefaults`");
    DenoiseDefaults {
        steps: u32::try_from(
            defaults
                .get("steps")
                .and_then(Value::as_u64)
                .expect("`modelDefaults.steps`"),
        )
        .expect("`modelDefaults.steps` fits a u32"),
        sampler: defaults
            .get("sampler")
            .and_then(Value::as_str)
            .expect("`modelDefaults.sampler`")
            .to_string(),
        scheduler: defaults
            .get("scheduler")
            .and_then(Value::as_str)
            .expect("`modelDefaults.scheduler`")
            .to_string(),
        guidance: defaults
            .get("guidance")
            .and_then(Value::as_f64)
            .map(|g| g as f32),
    }
}

/// Intern a fixture-supplied menu so it can sit in a `Capabilities`, whose menus are `&'static str`
/// by design (they are compile-time descriptor data in production). Leaking is fine and bounded in
/// a test binary: one small allocation per fixture menu entry.
fn intern(values: Option<&Vec<Value>>) -> Vec<&'static str> {
    values
        .map(|values| {
            values
                .iter()
                .map(|value| {
                    let owned = value
                        .as_str()
                        .expect("menu entries are strings")
                        .to_string();
                    let leaked: &'static str = Box::leak(owned.into_boxed_str());
                    leaked
                })
                .collect()
        })
        .unwrap_or_default()
}

/// The `Capabilities` a fixture's `capabilities` block describes, padded with the ordinary bounds an
/// image model advertises so the shared request floor has something real to check against.
fn capabilities_of(fixture: &Value) -> Capabilities {
    let caps = fixture.get("capabilities");
    Capabilities {
        supports_denoise_passes: caps
            .and_then(|c| c.get("supportsDenoisePasses"))
            .and_then(Value::as_bool)
            .unwrap_or(true),
        samplers: intern(
            caps.and_then(|c| c.get("samplers"))
                .and_then(Value::as_array),
        ),
        schedulers: intern(
            caps.and_then(|c| c.get("schedulers"))
                .and_then(Value::as_array),
        ),
        supports_guidance: true,
        supports_negative_prompt: true,
        supports_lora: true,
        // The per-family denoise-pass surface (sc-20425). Both keys are optional and default
        // **permissive**, exactly as `supportsDenoisePasses` above does: a fixture describes a
        // request shape, and one written before this capability existed must not be retroactively
        // invalidated by it. A fixture that wants the rejection asks for it by name.
        denoise_pass_surface: DenoisePassSurface {
            native_schedulers: intern_static(
                caps.and_then(|c| c.get("nativeSchedulers"))
                    .and_then(Value::as_array),
            ),
            per_pass_adapters: caps
                .and_then(|c| c.get("perPassAdapters"))
                .and_then(Value::as_bool)
                .unwrap_or(true),
        },
        max_count: 8,
        min_size: 64,
        max_size: 4096,
        ..Default::default()
    }
}

/// [`intern`] for the `&'static [&'static str]` shape [`DenoisePassSurface`] carries.
fn intern_static(values: Option<&Vec<Value>>) -> &'static [&'static str] {
    Box::leak(intern(values).into_boxed_slice())
}

fn loaded_adapters_of(fixture: &Value) -> Option<usize> {
    fixture
        .get("capabilities")
        .and_then(|c| c.get("loadedAdapters"))
        .and_then(Value::as_u64)
        .map(|n| usize::try_from(n).expect("loadedAdapters fits a usize"))
}

/// The raw `advanced.denoisePasses` array a fixture carries, if any.
fn raw_passes(fixture: &Value) -> Option<&Value> {
    fixture
        .get("request")
        .and_then(|r| r.get("advanced"))
        .and_then(|a| a.get("denoisePasses"))
        .filter(|v| !v.is_null())
}

/// Decode the fixture request into a `GenerationRequest`.
///
/// `advanced.phases` is inspected for presence only — the Krea RAW multi-phase payload has its own
/// contract and is deliberately not decoded here (see the fixture README).
fn request_of(fixture: &Value) -> Result<GenerationRequest, DenoisePassError> {
    let request = fixture.get("request").expect("fixture declares `request`");
    let advanced = request
        .get("advanced")
        .cloned()
        .unwrap_or_else(|| Value::Object(Default::default()));
    let phases = advanced
        .get("phases")
        .and_then(Value::as_array)
        .map(|phases| vec![GenerationPhase::default(); phases.len()]);
    // Only `denoisePasses` participates in the strict decode; `advanced` legitimately carries other
    // keys (`strength`, …) that flatten onto other request fields. A *null* `advanced` is passed
    // through verbatim rather than isolated into an empty object, so a fixture declaring that shape
    // genuinely exercises the null-block branch on both sides of the contract.
    let denoise_passes = if advanced.is_null() {
        denoise_passes_from_advanced_json(&advanced)?
    } else {
        let mut isolated = serde_json::Map::new();
        if let Some(passes) = advanced.get("denoisePasses") {
            isolated.insert("denoisePasses".to_string(), passes.clone());
        }
        denoise_passes_from_advanced_json(&Value::Object(isolated))?
    };
    Ok(GenerationRequest {
        prompt: "fixture".to_string(),
        width: 1024,
        height: 1024,
        seed: request.get("seed").and_then(Value::as_u64),
        steps: request
            .get("steps")
            .and_then(Value::as_u64)
            .map(|s| u32::try_from(s).expect("steps fits a u32")),
        sampler: request
            .get("sampler")
            .and_then(Value::as_str)
            .map(str::to_string),
        scheduler: request
            .get("scheduler")
            .and_then(Value::as_str)
            .map(str::to_string),
        guidance: request
            .get("guidance")
            .and_then(Value::as_f64)
            .map(|g| g as f32),
        phases,
        denoise_passes,
        ..Default::default()
    })
}

fn context_of<'a>(caps: &'a Capabilities, fixture: &Value) -> DenoisePassContext<'a> {
    caps.denoise_pass_context(loaded_adapters_of(fixture))
}

// =================================================================================================

/// Every valid fixture decodes, re-encodes to its canonical form byte-for-byte, and resolves to the
/// checked-in plan — the inference half of AC 4.
#[test]
fn valid_fixtures_round_trip_and_resolve() {
    for path in json_files(&fixture_root().join("valid")) {
        let name = stem(&path);
        let fixture = read_json(&path);
        assert_eq!(
            fixture.get("fixture").and_then(Value::as_str),
            Some(name.as_str()),
            "{name}: the `fixture` field must equal the file stem"
        );

        let request = request_of(&fixture)
            .unwrap_or_else(|err| panic!("{name}: a valid fixture must decode, got {err}"));

        // Round-trip: decode → encode reproduces the canonical array exactly (tolerance 0.0). A
        // fixture that writes explicit nulls declares its normalized form in
        // `canonicalDenoisePasses`; everything else is already canonical.
        if let Some(passes) = &request.denoise_passes {
            let expected = fixture
                .get("canonicalDenoisePasses")
                .or_else(|| raw_passes(&fixture))
                .unwrap_or_else(|| panic!("{name}: fixture carries denoise passes"));
            let encoded = denoise_passes_to_json(passes);
            assert!(
                json_equal_within(&encoded, expected, 0.0),
                "{name}: re-encoding is not lossless\n  encoded: {encoded}\n expected: {expected}"
            );
            // ... and decoding the encoding is a fixed point.
            let redecoded = gen_core::denoise_passes_from_json(&encoded)
                .unwrap_or_else(|err| panic!("{name}: canonical form must decode, got {err}"));
            assert_eq!(
                &redecoded, passes,
                "{name}: decode∘encode is not the identity"
            );
        } else {
            assert!(
                raw_passes(&fixture).is_none(),
                "{name}: decoded to no passes but the fixture carries a `denoisePasses` array"
            );
        }

        // The shared request floor accepts it.
        let caps = capabilities_of(&fixture);
        caps.validate_request(&name, &request)
            .unwrap_or_else(|err| panic!("{name}: the shared floor must accept it, got {err}"));

        // ... and it resolves to the checked-in plan.
        let job_seed = request.seed.expect("fixtures name an explicit job seed");
        let plan = request
            .resolve_denoise_plan(
                job_seed,
                &defaults_of(&fixture),
                &context_of(&caps, &fixture),
            )
            .unwrap_or_else(|err| panic!("{name}: must resolve, got {err}"));
        let expected_path = fixture_root()
            .join("resolved")
            .join(format!("{name}.plan.json"));
        let expected = read_json(&expected_path);
        assert!(
            json_equal_within(&plan.to_json(), &expected, PLAN_TOLERANCE),
            "{name}: resolved plan differs\n  actual: {}\nexpected: {expected}",
            plan.to_json()
        );
        // The expectation file itself must be a decodable plan, so the SceneWorks side can consume
        // it as a plan rather than only diff it as text.
        assert_eq!(
            ResolvedDenoisePlan::from_json(&expected)
                .unwrap_or_else(|err| panic!("{name}: plan fixture must decode, got {err}")),
            plan,
            "{name}: plan fixture decodes to a different plan"
        );
    }
}

/// Every invalid fixture is rejected **before execution**, on the pass index and field it declares
/// — AC 5.
#[test]
fn invalid_fixtures_are_rejected_on_the_declared_pass_and_field() {
    for path in json_files(&fixture_root().join("invalid")) {
        let name = stem(&path);
        let fixture = read_json(&path);
        assert_eq!(
            fixture.get("fixture").and_then(Value::as_str),
            Some(name.as_str()),
            "{name}: the `fixture` field must equal the file stem"
        );
        let expected = fixture
            .get("expectedError")
            .unwrap_or_else(|| panic!("{name}: an invalid fixture declares `expectedError`"));
        let expected_index = expected
            .get("passIndex")
            .and_then(|v| {
                if v.is_null() {
                    None
                } else {
                    Some(v.as_u64().expect("passIndex is an integer or null"))
                }
            })
            .map(|i| usize::try_from(i).expect("passIndex fits a usize"));
        let expected_field = expected
            .get("field")
            .and_then(Value::as_str)
            .expect("`expectedError.field`");
        let expected_issue = expected
            .get("issue")
            .and_then(Value::as_str)
            .expect("`expectedError.issue`");

        // A fixture fails either at DECODE (malformed JSON shape) or at VALIDATION; both produce the
        // same typed error, so the harness accepts whichever comes first.
        let caps = capabilities_of(&fixture);
        let err = match request_of(&fixture) {
            Err(err) => err,
            Ok(request) => {
                let passes = request.denoise_passes.clone().unwrap_or_default();
                let err = validate_denoise_passes(
                    &passes,
                    request.phases.is_some(),
                    &context_of(&caps, &fixture),
                )
                .expect_err(&format!("{name}: must be rejected"));
                // The shared request floor must reject it too — a typed contract error that only
                // the standalone validator catches is a contract the floor does not enforce.
                //
                // The one exception a fixture may declare is `checkedBy: "model"`: the floor holds
                // only a `Capabilities`, never a `LoadSpec`, so it cannot know how many adapters
                // were provisioned and the per-pass adapter index bound is the model's to check
                // when it resolves the plan against its real stack. Declaring that in the fixture
                // (rather than silently skipping) is what tells the SceneWorks side the same.
                let checked_by = expected
                    .get("checkedBy")
                    .and_then(Value::as_str)
                    .unwrap_or("floor");
                match checked_by {
                    "floor" => {
                        let floor = caps
                            .validate_request(&name, &request)
                            .expect_err(&format!("{name}: the shared floor must reject it"));
                        match (&floor, err.is_capability_gap()) {
                            (Error::Unsupported(_), true) | (Error::Msg(_), false) => {}
                            other => panic!(
                                "{name}: floor error kind does not match the issue class: {other:?}"
                            ),
                        }
                        assert!(
                            floor.to_string().contains(&err.path()),
                            "{name}: the floor's message must carry the path {}\n  got: {floor}",
                            err.path()
                        );
                    }
                    "model" => {
                        caps.validate_request(&name, &request).unwrap_or_else(|e| {
                            panic!(
                                "{name}: declared `checkedBy: model`, so the model-free floor must \
                                 pass it, got {e}"
                            )
                        });
                    }
                    other => panic!("{name}: unknown `checkedBy` value {other:?}"),
                }
                err
            }
        };

        assert_eq!(
            err.pass_index(),
            expected_index,
            "{name}: pass index — {err}"
        );
        assert_eq!(err.field().name(), expected_field, "{name}: field — {err}");
        assert_eq!(err.issue().code(), expected_issue, "{name}: issue — {err}");
    }
}

/// Every valid fixture has a checked-in plan expectation, and no orphan expectation exists. Without
/// this, adding a fixture and forgetting its plan silently reduces coverage on both sides.
#[test]
fn valid_fixtures_and_plan_expectations_are_in_bijection() {
    let mut valid: Vec<String> = json_files(&fixture_root().join("valid"))
        .iter()
        .map(|p| stem(p))
        .collect();
    let mut resolved: Vec<String> = json_files(&fixture_root().join("resolved"))
        .iter()
        .map(|p| stem(p))
        .collect();
    valid.sort();
    resolved.sort();
    assert_eq!(valid, resolved, "valid/ and resolved/ must be in bijection");
}

/// AC 1 + AC 2: a request serialized before sc-20415 carries no `denoisePasses`, still deserializes,
/// and resolves to exactly the plan an explicit one-pass request resolves to.
#[test]
fn a_legacy_request_resolves_exactly_like_an_explicit_single_pass() {
    let legacy = read_json(
        &fixture_root()
            .join("valid")
            .join("legacy_request_without_denoise_passes.json"),
    );
    let explicit = read_json(
        &fixture_root()
            .join("valid")
            .join("single_pass_inherits_all.json"),
    );

    let legacy_request = request_of(&legacy).expect("the legacy fixture decodes");
    assert!(
        legacy_request.denoise_passes.is_none(),
        "the legacy fixture must decode to the absent state, not an empty list"
    );
    assert!(
        raw_passes(&legacy).is_none(),
        "the legacy fixture must not carry a `denoisePasses` key at all"
    );

    let explicit_request = request_of(&explicit).expect("the explicit fixture decodes");
    let caps = capabilities_of(&legacy);
    let legacy_plan = legacy_request
        .resolve_denoise_plan(
            legacy_request.seed.expect("seed"),
            &defaults_of(&legacy),
            &context_of(&caps, &legacy),
        )
        .expect("the legacy request resolves");
    let explicit_plan = explicit_request
        .resolve_denoise_plan(
            explicit_request.seed.expect("seed"),
            &defaults_of(&explicit),
            &context_of(&caps, &explicit),
        )
        .expect("the explicit request resolves");
    assert_eq!(
        legacy_plan, explicit_plan,
        "a one-pass plan must resolve equivalently to the top-level request (AC 3)"
    );
    assert!(legacy_plan.is_single_pass());
    assert_eq!(
        legacy_plan.passes[0].seed,
        legacy_request.seed.expect("seed"),
        "pass 0 must reuse the job seed verbatim, or every persisted replay breaks"
    );
}

/// A request with no denoise passes must be untouched by every part of the new surface — the
/// additive guarantee behind AC 1.
#[test]
fn requests_without_denoise_passes_are_unaffected() {
    let bare = GenerationRequest {
        prompt: "a cat".to_string(),
        steps: Some(20),
        guidance: Some(3.5),
        sampler: Some("euler".to_string()),
        scheduler: Some("normal".to_string()),
        ..Default::default()
    };
    assert!(bare.denoise_passes.is_none(), "the Default is absent");
    assert!(bare.first_nonfinite_float().is_none());
    let caps = Capabilities {
        supports_guidance: true,
        samplers: vec!["euler"],
        schedulers: vec!["normal"],
        max_count: 8,
        min_size: 64,
        max_size: 4096,
        ..Default::default()
    };
    // Note `supports_denoise_passes` is false here — the capability gate must not fire for a request
    // that does not use the feature.
    assert!(!caps.supports_denoise_passes);
    caps.validate_request("legacy-model", &bare)
        .expect("an untouched legacy request still validates");

    // ... and it still resolves through the shared resolver, so a provider can adopt the plan
    // unconditionally instead of branching on the field's presence.
    let plan = bare
        .resolve_denoise_plan(
            99,
            &DenoiseDefaults::new(28, "dpmpp_2m", "karras"),
            &caps.denoise_pass_context(None),
        )
        .expect("resolves");
    assert!(plan.is_single_pass());
    assert_eq!(plan.passes[0].steps, 20);
    assert_eq!(plan.passes[0].sampler, "euler");
    assert_eq!(plan.passes[0].seed, 99);
}

/// The finiteness floor reaches into the pass list even for a provider that bypasses
/// `validate_request` and calls `ensure_finite_floats` directly (the flux1 carve-out shape).
#[test]
fn the_finiteness_floor_covers_pass_floats() {
    for (make, key) in [
        (
            (|| DenoisePass {
                denoise: Some(f32::NAN),
                ..Default::default()
            }) as fn() -> DenoisePass,
            "denoisePasses.denoise",
        ),
        (
            || DenoisePass {
                guidance: Some(f32::INFINITY),
                ..Default::default()
            },
            "denoisePasses.guidance",
        ),
        (
            || DenoisePass {
                adapters: vec![gen_core::PhaseAdapter {
                    adapter: 0,
                    weight: Some(f32::NEG_INFINITY),
                }],
                ..Default::default()
            },
            "denoisePasses.adapter.weight",
        ),
    ] {
        let req = GenerationRequest {
            denoise_passes: Some(vec![make()]),
            ..Default::default()
        };
        assert_eq!(
            req.first_nonfinite_float().map(|(field, _)| field),
            Some(key)
        );
        assert!(req.ensure_finite_floats().is_err());
    }
}
