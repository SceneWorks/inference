// Opt-in successful learned-output observation; default expansion is the original expression.
macro_rules! __model_proof_identity {
    ($value:expr) => {
        ($value)
    };
}
const _: () = __model_proof_identity!(());

#[cfg(feature = "model-inference-proof")]
macro_rules! __model_proof_select {
    ($original:expr, $observed:expr) => {
        ($observed)
    };
}
#[cfg(not(feature = "model-inference-proof"))]
macro_rules! __model_proof_select {
    ($original:expr, $observed:expr) => {
        ($original)
    };
}
const _: () = __model_proof_select!((), ());

#[cfg(feature = "model-inference-proof")]
macro_rules! __model_proof_inspect {
    ($value:expr, $observe:expr) => {
        crate::model_proof::inspect_successful_output!($value, $observe)
    };
}
#[cfg(not(feature = "model-inference-proof"))]
macro_rules! __model_proof_inspect {
    ($value:expr, $observe:expr) => {
        __model_proof_identity!($value)
    };
}

#[cfg(feature = "model-inference-proof")]
macro_rules! __model_proof_device {
    ($tensor:expr) => {{
        if let Some(__scope) = crate::model_proof::requested_scope() {
            let __tensor = $tensor;
            let __device = __tensor.device();
            let __backend = if __device.is_cuda() {
                "candle-cuda"
            } else if __device.is_metal() {
                "candle-metal"
            } else if __device.is_cpu() {
                "candle-cpu"
            } else {
                "unknown"
            };
            crate::model_proof::emit_observed_tensor_backend(&__scope, __backend);
        }
    }};
}

#[cfg(feature = "model-inference-proof")]
macro_rules! __model_proof_output {
    ($value:expr) => {
        __model_proof_inspect!($value, |__output| {
            __model_proof_device!(__output);
        })
    };
    ($value:expr, optional) => {
        __model_proof_inspect!($value, |__output| {
            if let Some(__tensor) = __output.as_ref() {
                __model_proof_device!(__tensor);
            }
        })
    };
    ($value:expr, tensors) => {
        __model_proof_inspect!($value, |__output| {
            for __tensor in __output.iter() {
                __model_proof_device!(__tensor);
            }
        })
    };
    ($value:expr, pair) => {
        __model_proof_inspect!($value, |__output| {
            __model_proof_device!(&__output.0);
            __model_proof_device!(&__output.1);
        })
    };
    ($value:expr, triple) => {
        __model_proof_inspect!($value, |__output| {
            __model_proof_device!(&__output.0);
            __model_proof_device!(&__output.1);
            __model_proof_device!(&__output.2);
        })
    };
    ($value:expr, first) => {
        __model_proof_inspect!($value, |__output| {
            __model_proof_device!(&__output.0);
        })
    };
}

#[cfg(not(feature = "model-inference-proof"))]
macro_rules! __model_proof_output {
    ($value:expr $(, $mode:ident)?) => {
        __model_proof_inspect!($value, |_| {})
    };
}

#[cfg(any(test, feature = "model-inference-proof"))]
mod protocol {
    // Opt-in Candle model-proof protocol support. DEVICE callers must observe a
    // successful learned Tensor boundary, before host copying or postprocessing.
    // BODY markers belong to the actual test's original success tail, not here.

    use std::collections::HashSet;
    use std::sync::{Mutex, OnceLock};

    #[derive(Default)]
    struct Observations {
        seen: HashSet<(String, &'static str)>,
    }

    impl Observations {
        fn observe(
            &mut self,
            scope: Option<&str>,
            backend: &'static str,
            emit: &mut impl FnMut(String),
        ) {
            let Some(scope) = scope else { return };
            if !matches!(backend, "candle-cpu" | "candle-metal" | "candle-cuda") {
                return;
            }
            if self.seen.insert((scope.to_owned(), backend)) {
                emit(format!("\nMODEL_INFERENCE_DEVICE {scope} {backend}"));
            }
        }
    }

    static OBSERVATIONS: OnceLock<Mutex<Observations>> = OnceLock::new();

    pub fn requested_scope() -> Option<String> {
        std::env::var("MODEL_INFERENCE_CASE_FQN").ok()
    }

    // The backend is obtained by the source-bound observer macro from the actual
    // successful learned output's Device properties. No ambient/default device is
    // selected, created, or evaluated by this helper.
    pub fn emit_observed_tensor_backend(scope: &str, backend: &'static str) {
        let mut lines = Vec::new();
        OBSERVATIONS
            .get_or_init(|| Mutex::new(Observations::default()))
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .observe(Some(scope), backend, &mut |line| lines.push(line));
        for line in lines {
            println!("{line}");
        }
    }

    pub fn missing_required_work(reason: &str) {
        if let Some(scope) = requested_scope() {
            eprintln!("\nMODEL_INFERENCE_PREREQUISITE_MISSING {scope} {reason}");
        }
    }

    // Keep the optional diagnostic interface type-checked without a lint allowance
    // or any execution when a particular including target has no asset-skip path.
    const _: fn(&str) = missing_required_work;
    const _: fn(&str, &'static str) = emit_observed_tensor_backend;

    macro_rules! inspect_successful_output {
        ($value:expr, $observe:expr) => {
            ($value).inspect($observe)
        };
    }
    pub(crate) use inspect_successful_output;

    #[cfg(test)]
    mod observer_tests {
        use super::*;
        use std::cell::{Cell, RefCell};
        use std::rc::Rc;

        #[test]
        fn formatter_single_expression_match_and_closure_blocks_keep_temporary_drop_order() {
            struct Traced<'a>(&'static str, &'a RefCell<Vec<&'static str>>);
            impl Traced<'_> {
                fn produce(&self, ok: bool) -> Result<String, &'static str> {
                    self.1.borrow_mut().push("produce");
                    if ok {
                        Ok(String::from("owned"))
                    } else {
                        Err("same-error")
                    }
                }
            }
            impl Drop for Traced<'_> {
                fn drop(&mut self) {
                    self.1.borrow_mut().push(self.0);
                }
            }
            fn original_match(
                trace: &RefCell<Vec<&'static str>>,
                arm: bool,
            ) -> Result<String, &'static str> {
                let _local = Traced("drop-local", trace);
                match arm {
                    true => Traced("drop-receiver", trace).produce(true),
                    false => Traced("drop-receiver", trace).produce(false),
                }
            }
            fn observed_match(
                trace: &RefCell<Vec<&'static str>>,
                arm: bool,
            ) -> Result<String, &'static str> {
                let _local = Traced("drop-local", trace);
                match arm {
                    true => {
                        crate::model_proof::inspect_successful_output!(
                            Traced("drop-receiver", trace).produce(true),
                            |_| {}
                        )
                    }
                    false => {
                        crate::model_proof::inspect_successful_output!(
                            Traced("drop-receiver", trace).produce(false),
                            |_| {}
                        )
                    }
                }
            }
            for arm in [false, true] {
                let plain = RefCell::new(Vec::new());
                let observed = RefCell::new(Vec::new());
                assert_eq!(original_match(&plain, arm), observed_match(&observed, arm));
                assert_eq!(*observed.borrow(), *plain.borrow());
                let plain = RefCell::new(Vec::new());
                let observed = RefCell::new(Vec::new());
                let original_closure = || Traced("drop-receiver", &plain).produce(arm);
                let observed_closure = || {
                    crate::model_proof::inspect_successful_output!(
                        Traced("drop-receiver", &observed).produce(arm),
                        |_| {}
                    )
                };
                assert_eq!(original_closure(), observed_closure());
                assert_eq!(*observed.borrow(), *plain.borrow());
            }
        }

        #[test]
        fn source_bound_variant_evaluates_exactly_one_original_or_observed_branch() {
            let calls = Cell::new(0);
            let value = __model_proof_select!(
                {
                    calls.set(calls.get() + 1);
                    "original"
                },
                {
                    calls.set(calls.get() + 10);
                    "observed"
                }
            );
            if cfg!(feature = "model-inference-proof") {
                assert_eq!((value, calls.get()), ("observed", 10));
            } else {
                assert_eq!((value, calls.get()), ("original", 1));
            }
        }

        #[test]
        fn borrowed_noop_callback_delegation_preserves_original_flow_and_local_drops() {
            struct Lease<'a>(&'a RefCell<Vec<&'static str>>);
            impl Drop for Lease<'_> {
                fn drop(&mut self) {
                    self.0.borrow_mut().push("drop-lease");
                }
            }
            fn original(
                input: &str,
                trace: &RefCell<Vec<&'static str>>,
            ) -> Result<String, &'static str> {
                let _lease = Lease(trace);
                trace.borrow_mut().push("enter");
                if input.is_empty() {
                    return Err("original-error");
                }
                let output = input.to_owned();
                trace.borrow_mut().push("computed");
                trace.borrow_mut().push("readback");
                Ok(output)
            }
            fn using(
                input: &str,
                trace: &RefCell<Vec<&'static str>>,
                observe: impl FnOnce(&str),
            ) -> Result<String, &'static str> {
                let _lease = Lease(trace);
                trace.borrow_mut().push("enter");
                if input.is_empty() {
                    return Err("original-error");
                }
                let output = input.to_owned();
                trace.borrow_mut().push("computed");
                observe(&output);
                trace.borrow_mut().push("readback");
                Ok(output)
            }
            for input in ["", "borrowed-input"] {
                let plain = RefCell::new(Vec::new());
                let observed = RefCell::new(Vec::new());
                let events = RefCell::new(Vec::new());
                let expected = original(input, &plain);
                let actual = using(input, &observed, |value| {
                    events.borrow_mut().push(value.to_owned())
                });
                assert_eq!(actual, expected);
                assert_eq!(*observed.borrow(), *plain.borrow());
                assert_eq!(plain.borrow().last(), Some(&"drop-lease"));
                assert_eq!(events.borrow().len(), usize::from(!input.is_empty()));
                let no_op = RefCell::new(Vec::new());
                assert_eq!(using(input, &no_op, |_| {}), expected);
                assert_eq!(*no_op.borrow(), *plain.borrow());
            }
        }

        #[test]
        fn scope_and_backend_dedup_preserves_root_and_each_child() {
            let mut observations = Observations::default();
            let mut events = Vec::new();
            for scope in ["root", "root#child=0", "root#child=1", "other"] {
                for _ in 0..2 {
                    observations.observe(Some(scope), "candle-cpu", &mut |line| events.push(line));
                }
            }
            observations.observe(Some("root#child=0"), "candle-metal", &mut |line| {
                events.push(line)
            });
            observations.observe(Some("root#child=0"), "candle-cuda", &mut |line| {
                events.push(line)
            });
            assert_eq!(
                events,
                [
                    "\nMODEL_INFERENCE_DEVICE root candle-cpu",
                    "\nMODEL_INFERENCE_DEVICE root#child=0 candle-cpu",
                    "\nMODEL_INFERENCE_DEVICE root#child=1 candle-cpu",
                    "\nMODEL_INFERENCE_DEVICE other candle-cpu",
                    "\nMODEL_INFERENCE_DEVICE root#child=0 candle-metal",
                    "\nMODEL_INFERENCE_DEVICE root#child=0 candle-cuda",
                ]
            );
        }

        #[test]
        fn absent_request_does_not_consume_a_later_scope_observation() {
            let mut observations = Observations::default();
            let mut events = Vec::new();
            observations.observe(None, "candle-cpu", &mut |line| events.push(line));
            observations.observe(Some("root"), "unknown", &mut |line| events.push(line));
            assert!(events.is_empty());
            observations.observe(Some("root"), "candle-cpu", &mut |line| events.push(line));
            assert_eq!(events, ["\nMODEL_INFERENCE_DEVICE root candle-cpu"]);
        }

        #[test]
        fn successful_boundary_is_evaluated_once_and_keeps_noncopy_output() {
            let evaluations = Cell::new(0);
            let observations = Cell::new(0);
            let result: Result<String, &str> = crate::model_proof::inspect_successful_output!(
                {
                    evaluations.set(evaluations.get() + 1);
                    Ok(String::from("original"))
                },
                |value| {
                    observations.set(observations.get() + 1);
                    assert_eq!(value, "original");
                }
            );
            assert_eq!(result.unwrap(), "original");
            assert_eq!((evaluations.get(), observations.get()), (1, 1));
        }

        #[test]
        fn errors_question_mark_and_panics_do_not_report_a_successful_boundary() {
            let observations = Cell::new(0);
            let run = || -> Result<(), &str> {
                crate::model_proof::inspect_successful_output!(
                    Err::<String, _>("original-error"),
                    |_| {
                        observations.set(observations.get() + 1);
                    }
                )?;
                panic!("question mark must preserve early error flow");
            };
            assert_eq!(run(), Err("original-error"));
            let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let panic_boundary = || -> Result<String, &str> { panic!("original-panic") };
                let _: Result<String, &str> =
                    crate::model_proof::inspect_successful_output!(panic_boundary(), |_| {
                        observations.set(observations.get() + 1);
                    });
            }));
            assert!(panic.is_err());
            assert_eq!(observations.get(), 0);
        }

        #[test]
        fn tuple_vec_and_inner_success_before_cancellation_remain_observable() {
            #[derive(Debug, PartialEq)]
            struct MockOutput {
                compute_backend: &'static str,
            }
            let observed = RefCell::new(Vec::new());
            let tuple: Result<(MockOutput, Vec<MockOutput>), ()> = crate::model_proof::inspect_successful_output!(
                Ok((
                    MockOutput {
                        compute_backend: "candle-cuda"
                    },
                    vec![
                        MockOutput {
                            compute_backend: "candle-cpu"
                        },
                        MockOutput {
                            compute_backend: "candle-metal"
                        },
                    ]
                )),
                |outputs| {
                    observed.borrow_mut().push(outputs.0.compute_backend);
                    for output in &outputs.1 {
                        observed.borrow_mut().push(output.compute_backend);
                    }
                }
            );
            assert!(tuple.is_ok());
            assert_eq!(
                &*observed.borrow(),
                &["candle-cuda", "candle-cpu", "candle-metal"]
            );
            observed.borrow_mut().clear();
            let cancelled_api = || -> Result<Option<Vec<f32>>, ()> {
                let _computed = crate::model_proof::inspect_successful_output!(
                    Ok::<_, ()>(MockOutput {
                        compute_backend: "candle-cuda"
                    }),
                    |output| observed.borrow_mut().push(output.compute_backend)
                )?;
                Ok(None)
            };
            assert_eq!(cancelled_api(), Ok(None));
            assert_eq!(&*observed.borrow(), &["candle-cuda"]);
        }

        #[cfg(not(feature = "model-inference-proof"))]
        #[test]
        fn feature_off_wrapper_is_the_original_expression_and_temporary_scope() {
            struct Owner {
                value: String,
                dropped: Rc<Cell<bool>>,
            }
            impl Drop for Owner {
                fn drop(&mut self) {
                    self.dropped.set(true);
                }
            }
            let evaluations = Cell::new(0);
            let dropped = Rc::new(Cell::new(false));
            // This borrow extends Owner's lifetime exactly as the original direct
            // field expression does. A block-local result temporary would fail it.
            let borrowed = &__model_proof_output!(
                {
                    evaluations.set(evaluations.get() + 1);
                    Owner {
                        value: String::from("kept"),
                        dropped: dropped.clone(),
                    }
                }
                .value
            );
            assert_eq!(borrowed, "kept");
            assert_eq!(evaluations.get(), 1);
            assert!(!dropped.get());
            let error: Result<String, &str> = __model_proof_output!(Err("unchanged"));
            assert_eq!(error, Err("unchanged"));
        }

        #[test]
        fn noncopy_output_drop_order_matches_the_unobserved_call() {
            struct Output(Rc<RefCell<Vec<&'static str>>>);
            impl Drop for Output {
                fn drop(&mut self) {
                    self.0.borrow_mut().push("drop-output");
                }
            }
            let plain = Rc::new(RefCell::new(Vec::new()));
            {
                let value: Result<Output, ()> = Ok(Output(plain.clone()));
                plain.borrow_mut().push("map");
                // Keep the independently constructed owned Result alive to the
                // original block boundary, without an identity-map combinator.
                let _kept = value;
                plain.borrow_mut().push("end-body");
            }
            let observed = Rc::new(RefCell::new(Vec::new()));
            {
                let value: Result<Output, ()> = crate::model_proof::inspect_successful_output!(
                    Ok(Output(observed.clone())),
                    |_| {}
                );
                observed.borrow_mut().push("map");
                let _kept = value;
                observed.borrow_mut().push("end-body");
            }
            assert_eq!(&*plain.borrow(), &["map", "end-body", "drop-output"]);
            assert_eq!(*observed.borrow(), *plain.borrow());
        }

        #[test]
        fn borrowed_values_and_temporary_drop_scope_match_the_original_expression() {
            struct Owner(Rc<RefCell<Vec<&'static str>>>);
            impl Owner {
                fn produce(&self) -> Result<String, &'static str> {
                    self.0.borrow_mut().push("produce");
                    Ok(String::from("kept"))
                }
            }
            impl Drop for Owner {
                fn drop(&mut self) {
                    self.0.borrow_mut().push("drop-owner");
                }
            }
            let plain = Rc::new(RefCell::new(Vec::new()));
            let plain_value = Owner(plain.clone()).produce().inspect(|_| {
                plain.borrow_mut().push("after-call");
            });
            assert_eq!(&*plain.borrow(), &["produce", "after-call", "drop-owner"]);
            let observed = Rc::new(RefCell::new(Vec::new()));
            let value = crate::model_proof::inspect_successful_output!(
                Owner(observed.clone()).produce(),
                |output| {
                    assert_eq!(output, "kept");
                }
            )
            .inspect(|_| {
                observed.borrow_mut().push("after-call");
            });
            assert_eq!(value, plain_value);
            assert_eq!(*observed.borrow(), *plain.borrow());
            let text = String::from("borrowed");
            let borrowed: Result<&str, ()> =
                crate::model_proof::inspect_successful_output!(Ok(text.as_str()), |value| {
                    assert_eq!(*value, "borrowed");
                });
            assert_eq!(borrowed.unwrap(), text);
            assert_eq!(__model_proof_identity!(text.as_str()), "borrowed");
        }
    }
}

#[cfg(any(test, feature = "model-inference-proof"))]
pub(crate) use protocol::inspect_successful_output;
#[cfg(feature = "model-inference-proof")]
pub(crate) use protocol::{emit_observed_tensor_backend, missing_required_work, requested_scope};
#[cfg(feature = "model-inference-proof")]
const _: fn() -> Option<String> = requested_scope;
#[cfg(feature = "model-inference-proof")]
const _: fn(&str, &'static str) = emit_observed_tensor_backend;
#[cfg(feature = "model-inference-proof")]
const _: fn(&str) = missing_required_work;
