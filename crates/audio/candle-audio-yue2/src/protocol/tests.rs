//! Protocol parity with the pinned upstream (`protocol_cases.json`, `tokenizer_synthetic.json`).

use serde_json::{json, Value};

use super::*;
use crate::test_fixtures::{float_of, ids, json as fixture, same, synthetic};

fn outcome<T>(case: &Value, result: &Result<T, ProtocolError>) -> bool {
    let expect_ok = case.get("ok").is_some();
    assert_ne!(
        expect_ok,
        case.get("error").is_some(),
        "fixture case must be ok xor error: {case}"
    );
    if expect_ok != result.is_ok() {
        panic!(
            "case {case}: upstream {}, native {}",
            if expect_ok { "accepts" } else { "refuses" },
            match result {
                Ok(_) => "accepts".to_string(),
                Err(e) => format!("refuses ({e})"),
            }
        );
    }
    expect_ok
}

/// One `{"float": ..}` marker field goes through the typed API (JSON cannot carry non-finite
/// numbers); everything else goes through the JSON entry point.
fn resolve_sampling(default: &Sampling, overrides: &Value) -> Result<Sampling, ProtocolError> {
    let object = overrides.as_object().unwrap();
    let marker = object.iter().find_map(|(k, v)| float_of(v).map(|f| (k, f)));
    match marker {
        None => default.with_json_overrides(overrides),
        Some((key, value)) => {
            assert_eq!(object.len(), 1, "marker cases carry one field");
            let mut o = SamplingOverrides::default();
            match key.as_str() {
                "temperature" => o.temperature = Some(value),
                "top_p" => o.top_p = Some(value),
                "repetition_penalty" => o.repetition_penalty = Some(value),
                other => panic!("no float field {other}"),
            }
            default.with_overrides(&o)
        }
    }
}

/// Every sampling override case (both phases' defaults) is accepted or refused exactly as
/// upstream's `resolve_sampling`, with the same resolved values.
#[test]
fn sampling_validation_matches_upstream() {
    let cases = fixture("protocol_cases.json");
    let cases = cases["sampling"].as_array().unwrap();
    assert!(cases.len() >= 80, "fixture lost cases");
    for case in cases {
        let default = match case["phase"].as_str().unwrap() {
            "abc" => Sampling::abc_default(),
            _ => Sampling::semantic_default(),
        };
        let result = resolve_sampling(&default, &case["overrides"]);
        if outcome(case, &result) {
            let got = result.unwrap().to_json();
            assert!(same(&got, &case["ok"]), "case {case}: native {got}");
        }
    }
}

/// Non-finite values are refused for every float control, through the typed API too.
#[test]
fn non_finite_sampling_values_are_refused() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for o in [
            SamplingOverrides {
                temperature: Some(bad),
                ..Default::default()
            },
            SamplingOverrides {
                top_p: Some(bad),
                ..Default::default()
            },
            SamplingOverrides {
                repetition_penalty: Some(bad),
                ..Default::default()
            },
        ] {
            assert!(
                Sampling::semantic_default().with_overrides(&o).is_err(),
                "{o:?}"
            );
        }
    }
    let err = GenerationConfig::from_json(&json!({"semantic": {"temperature": true}})).unwrap_err();
    assert!(
        matches!(
            err,
            ProtocolError::Invalid {
                field: "temperature",
                ..
            }
        ),
        "{err}"
    );
}

#[test]
fn defaults_are_upstreams() {
    let cases = fixture("protocol_cases.json");
    let upstream_default = cases["generation_config"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["input"] == json!({}))
        .unwrap();
    assert!(same(
        &GenerationConfig::default().to_json(),
        &upstream_default["ok"]
    ));
}

/// `GenerationConfig.from_dict` parity: fixed context and midpoint method, positive integer
/// steps, per-phase overrides over the defaults, unknown keys refused.
#[test]
fn generation_config_validation_matches_upstream() {
    let cases = fixture("protocol_cases.json");
    for case in cases["generation_config"].as_array().unwrap() {
        let result = GenerationConfig::from_json(&case["input"]);
        if outcome(case, &result) {
            let got = result.unwrap().to_json();
            assert!(same(&got, &case["ok"]), "case {case}: native {got}");
        }
    }
}

fn song_request(input: &Value) -> Result<SongRequest, ProtocolError> {
    let object = input.as_object().unwrap();
    match object.get("cfg_scale").and_then(float_of) {
        None => SongRequest::from_json(input),
        Some(scale) => {
            let mut rest = object.clone();
            rest.remove("cfg_scale");
            let base = SongRequest::from_json(&Value::Object(rest))?;
            SongRequest::new(SongRequestSpec {
                cfg_scale: Some(scale),
                ..base.spec()
            })
        }
    }
}

/// Every `SongRequest` case — cot spelling, seed range and type, filename-safe id, external ABC
/// with cot=off or blank (Python whitespace), cfg range and finiteness, missing/unknown/mistyped
/// fields — is accepted or refused as upstream, with the same guidance and prompt text.
#[test]
fn request_validation_matches_upstream() {
    let cases = fixture("protocol_cases.json");
    let cases = cases["request"].as_array().unwrap();
    assert!(cases.len() >= 45, "fixture lost cases");
    for case in cases {
        let result = song_request(&case["input"]);
        if outcome(case, &result) {
            let request = result.unwrap();
            let ok = &case["ok"];
            if ok.get("request").is_none() {
                assert!(same(&request.to_json(), ok), "case {case}");
                continue;
            }
            assert!(same(&request.to_json(), &ok["request"]), "case {case}");
            assert_eq!(
                request.guidance(),
                ok["guidance"].as_f64().unwrap(),
                "case {case}"
            );
            assert_eq!(request.text(), ok["text"].as_str().unwrap(), "case {case}");
        }
    }
}

/// External ABC with cot=off is refused explicitly, and so is ABC Python considers blank —
/// including the ASCII separators U+001C..U+001F that Rust's `char::is_whitespace` misses.
#[test]
fn external_abc_rules() {
    let off = SongRequestSpec {
        cot: CotMode::Off,
        abc: Some("X:1".into()),
        ..SongRequestSpec::new("pop", "la")
    };
    assert!(matches!(
        SongRequest::new(off),
        Err(ProtocolError::ExternalAbc)
    ));
    let blank = SongRequestSpec {
        abc: Some("\u{1c}\u{1f} \n".into()),
        ..SongRequestSpec::new("pop", "la")
    };
    assert!(matches!(
        SongRequest::new(blank),
        Err(ProtocolError::ExternalAbc)
    ));
    let full = SongRequest::new(SongRequestSpec::new("pop", "la")).unwrap();
    let edited = full.with_abc("X:1\nK:C\nC D E F|").unwrap();
    assert_ne!(edited, full, "an edited ABC is a new request");
    assert_eq!(edited.abc(), Some("X:1\nK:C\nC D E F|"));
    let off = SongRequest::new(SongRequestSpec {
        cot: CotMode::Off,
        ..SongRequestSpec::new("pop", "la")
    })
    .unwrap();
    assert!(matches!(
        off.with_abc("X:1"),
        Err(ProtocolError::ExternalAbc)
    ));
}

/// `python_isspace` is exactly Python's `str.isspace` over every code point.
#[test]
fn python_isspace_matches_python_over_all_code_points() {
    let cases = fixture("protocol_cases.json");
    let expected: Vec<u32> = cases["python_isspace"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let native: Vec<u32> = (0..=0x10FFFFu32)
        .filter_map(char::from_u32)
        .filter(|&c| python_isspace(c))
        .map(u32::from)
        .collect();
    assert_eq!(native, expected);
}

/// `song_chunks` / `chunk_ranges` parity, including the exhausted-context refusals.
#[test]
fn chunk_ranges_match_upstream() {
    let cases = fixture("protocol_cases.json");
    for case in cases["chunk_ranges"].as_array().unwrap() {
        let arg = |k: &str| case[k].as_u64().unwrap() as usize;
        let result = chunk_ranges(arg("frames"), arg("prefix_tokens"), arg("context"));
        if outcome(case, &result) {
            let got: Vec<Value> = result
                .unwrap()
                .into_iter()
                .map(|(a, b)| json!([a, b]))
                .collect();
            assert_eq!(Value::from(got), case["ok"], "case {case}");
        } else if case["error"].as_str().unwrap().contains("acoustic context") {
            assert!(
                matches!(
                    chunk_ranges(arg("frames"), arg("prefix_tokens"), arg("context")),
                    Err(ProtocolError::ExhaustedAcousticContext { .. })
                ),
                "case {case}"
            );
        }
    }
}

/// `generate_tokens`' pre-prefill refusals: positive / negative budget past the context, and CFG
/// without a negative.
#[test]
fn generation_budget_matches_upstream() {
    let cases = fixture("protocol_cases.json");
    for case in cases["budget"].as_array().unwrap() {
        let sampling = Sampling::semantic_default()
            .with_overrides(&SamplingOverrides {
                max_tokens: case["max_tokens"].as_i64(),
                min_tokens: Some(0),
                ..Default::default()
            })
            .unwrap();
        let result = check_generation_budget(
            case["prefix_tokens"].as_u64().unwrap() as usize,
            case["negative_tokens"].as_u64().map(|n| n as usize),
            &sampling,
            case["cfg_scale"].as_f64().unwrap(),
        );
        outcome(case, &result);
    }
}

/// Positive, planner and negative prefixes for English and Chinese requests in all three cot
/// modes, with sampled and external ABC, CFG and NFD input, match upstream token for token; the
/// refusals (negative without ABC IDs, ABC IDs outside the ordinary vocabulary) match too.
#[test]
fn prefixes_match_upstream_for_every_mode() {
    let fixture = fixture("tokenizer_synthetic.json");
    let tok = synthetic();
    let cases = fixture["prefixes"].as_array().unwrap();
    let modes: std::collections::BTreeSet<&str> = cases
        .iter()
        .map(|c| c["request"]["cot"].as_str().unwrap())
        .collect();
    assert_eq!(modes.len(), 3, "every cot mode is covered");
    for case in cases {
        let request = SongRequest::from_json(&case["request"]).unwrap();
        assert_eq!(request.text(), case["text"].as_str().unwrap());
        assert_eq!(request.guidance(), case["guidance"].as_f64().unwrap());
        let check = |key: &str, got: Result<Vec<u32>, ProtocolError>| {
            let expected = &case[key];
            if outcome(expected, &got) {
                assert_eq!(got.unwrap(), ids(&expected["ok"]), "{} {key}", case["name"]);
            }
        };
        check("planner", token_prefixes(&request, tok, None));
        check("negative_without_abc", negative_prefix(&request, tok, None));
        if let Some(abc) = case.get("abc_ids") {
            let abc = ids(abc);
            check("positive", token_prefixes(&request, tok, Some(&abc)));
            check("negative", negative_prefix(&request, tok, Some(&abc)));
        }
    }
    for case in fixture["abc_id_cases"].as_array().unwrap() {
        let request = SongRequest::from_json(&case["request"]).unwrap();
        let abc = ids(&case["abc_ids"]);
        for (key, got) in [
            ("positive", token_prefixes(&request, tok, Some(&abc))),
            ("negative", negative_prefix(&request, tok, Some(&abc))),
        ] {
            if outcome(&case[key], &got) {
                assert_eq!(
                    got.unwrap(),
                    ids(&case[key]["ok"]),
                    "{} {key}",
                    case["name"]
                );
            }
        }
    }
}

/// The planner prefix ends at `<abc>`; a finished symbolic prefix closes the plan and opens the
/// music stream; cot=off skips the plan. (Shape, independent of any tokenizer table.)
#[test]
fn prefix_layout() {
    let tok = synthetic();
    let full = SongRequest::new(SongRequestSpec::new("pop", "la")).unwrap();
    let planner = token_prefixes(&full, tok, None).unwrap();
    assert_eq!((planner[0], *planner.last().unwrap()), (EOD, ABC_START));
    let done = token_prefixes(&full, tok, Some(&[1, 2, 3])).unwrap();
    assert_eq!(
        &done[done.len() - 6..],
        &[ABC_START, 1, 2, 3, ABC_END, MUSIC_START]
    );
    let off = SongRequest::new(SongRequestSpec {
        cot: CotMode::Off,
        ..SongRequestSpec::new("pop", "la")
    })
    .unwrap();
    let p = token_prefixes(&off, tok, None).unwrap();
    assert_eq!(&p[p.len() - 3..], &[ABC_START, ABC_END, MUSIC_START]);
    assert!(matches!(
        token_prefixes(&off, tok, Some(&[5])),
        Err(ProtocolError::AbcIdsWithoutPlan(1))
    ));
    assert_eq!(token_prefixes(&off, tok, Some(&[])).unwrap(), p);
    let neg = negative_prefix(&off, tok, None).unwrap();
    assert_eq!(*neg.last().unwrap(), MUSIC_START);
}
