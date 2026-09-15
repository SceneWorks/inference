//! SC-15811 Candle request-scoped residency conformance.
//!
//! This exercises the production shared owner through Candle's backend adapter: one warm generator
//! must serve warm → staged → warm requests, and mutating the request bit must change lifecycle.

use candle_gen::gen_core::CancelFlag;
use candle_gen::Residency;
use std::sync::{Arc, Mutex};

#[test]
fn one_warm_candle_owner_serves_warm_staged_warm_requests() {
    let loads = Arc::new(Mutex::new(Vec::new()));
    let text_loads = Arc::clone(&loads);
    let heavy_loads = Arc::clone(&loads);
    let residency = Residency::request_scoped(
        move |_| {
            text_loads.lock().unwrap().push("text");
            Ok(2u8)
        },
        move |_, _| {
            heavy_loads.lock().unwrap().push("heavy");
            Ok(3u8)
        },
    );

    let run = |stage_residency| {
        residency.run_request_scoped(
            stage_residency,
            false,
            &CancelFlag::new(),
            false,
            &mut |_| {},
            |text| Ok(*text),
            |_| Ok(()),
            |heavy, encoded, _| Ok(*heavy + encoded),
        )
    };

    assert_eq!(run(false).unwrap(), 5);
    assert_eq!(run(true).unwrap(), 5);
    assert_eq!(run(false).unwrap(), 5);
    assert_eq!(
        *loads.lock().unwrap(),
        vec!["text", "heavy", "text", "heavy", "text", "heavy"]
    );
}

#[test]
fn mutating_the_request_bit_changes_component_loads() {
    let loads = Arc::new(Mutex::new(0usize));
    let text_loads = Arc::clone(&loads);
    let heavy_loads = Arc::clone(&loads);
    let residency = Residency::request_scoped(
        move |_| {
            *text_loads.lock().unwrap() += 1;
            Ok(())
        },
        move |_, _| {
            *heavy_loads.lock().unwrap() += 1;
            Ok(())
        },
    );
    let run = |stage_residency| {
        residency.run_request_scoped(
            stage_residency,
            false,
            &CancelFlag::new(),
            false,
            &mut |_| {},
            |_| Ok(()),
            |_| Ok(()),
            |_, _, _| Ok(()),
        )
    };

    run(false).unwrap();
    let warm_loads = *loads.lock().unwrap();
    run(false).unwrap();
    assert_eq!(*loads.lock().unwrap(), warm_loads);
    run(true).unwrap();
    assert_eq!(*loads.lock().unwrap(), warm_loads + 2);
}
