//! Actual BoundGenerator lifecycle path, pure fake-provider delegates only.
//! No source facts are represented as native fence/runtime evidence.
use super::*;
use std::cell::RefCell;
use std::rc::Rc;

/// Native range calls are CPU-only and process-isolated: this cheap test must
/// not mutate another parallel test's global MLX default device. A child must
/// execute exactly one actual body; zero matches cannot count as validation.
fn cpu_test(name: &str, body: impl FnOnce()) {
    const CHILD: &str = "IOS_BOUND_NATIVE_LIFECYCLE_CPU_CHILD";
    if std::env::var(CHILD).ok().as_deref() == Some(name) {
        mlx_rs::Device::set_default(&mlx_rs::Device::cpu());
        mlx_gen::ios_quantum::with_cpu_diagnostic_headroom(1 << 30, || {
            body();
            Ok(())
        })
        .unwrap();
        return;
    }
    let fqn = format!("bound::lifecycle_tests::{name}");
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", &fqn, "--nocapture", "--test-threads=1"])
        .env(CHILD, name)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        output.status.success() && stdout.contains("1 passed; 0 failed"),
        "CPU lifecycle child {fqn} failed/zero matched: {}\n{stdout}\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
}

#[derive(Clone, Copy)]
enum Mode {
    Good,
    RefuseConfiguration,
    LegacyOnly,
    BodyError,
    Canceled,
}
#[derive(Default)]
struct Calls {
    begin: usize,
    mutable: usize,
    readonly: usize,
    body: usize,
    finish: Vec<MemoryRunOutcome>,
    request_address: usize,
    pixels_address: usize,
    selected_in_body: bool,
}
type Shared = Rc<RefCell<Calls>>;
struct Scope {
    calls: Shared,
    mode: Mode,
}
impl MemoryRequestScope for Scope {
    fn configure_request(&mut self, _: &mut GenerationRequest) -> Result<()> {
        self.calls.borrow_mut().mutable += 1;
        Ok(())
    }
    fn configure_admitted_request(&mut self, request: &GenerationRequest) -> Result<()> {
        let mut calls = self.calls.borrow_mut();
        calls.readonly += 1;
        calls.request_address = request as *const GenerationRequest as usize;
        calls.pixels_address = match request.conditioning.first() {
            Some(Conditioning::Reference { image, .. }) => image.pixels.as_ptr() as usize,
            _ => 0,
        };
        if matches!(self.mode, Mode::RefuseConfiguration) {
            Err(Error::Unsupported("actual configure refused".into()))
        } else {
            Ok(())
        }
    }
    fn enter_phase(&mut self, _: MemoryPhase) -> Result<()> {
        Ok(())
    }
    fn leave_phase(&mut self, _: MemoryPhase) -> Result<()> {
        Ok(())
    }
    fn configure_decode(&mut self, _: u32, _: u32, _: MemoryGeometry) -> Result<()> {
        Ok(())
    }
    fn configure_attention(&mut self, _: u32) -> Result<()> {
        Ok(())
    }
    fn materialize_transformer_window(&mut self, _: u32, _: u32) -> Result<()> {
        Ok(())
    }
    fn finish(&mut self, outcome: MemoryRunOutcome) -> Result<()> {
        self.calls.borrow_mut().finish.push(outcome);
        Ok(())
    }
}
struct LegacyScope(Scope);
impl MemoryRequestScope for LegacyScope {
    fn configure_request(&mut self, request: &mut GenerationRequest) -> Result<()> {
        self.0.configure_request(request)
    }
    // Deliberately no immutable override: exercise ACTUAL trait default.
    fn enter_phase(&mut self, p: MemoryPhase) -> Result<()> {
        self.0.enter_phase(p)
    }
    fn leave_phase(&mut self, p: MemoryPhase) -> Result<()> {
        self.0.leave_phase(p)
    }
    fn configure_decode(&mut self, e: u32, o: u32, g: MemoryGeometry) -> Result<()> {
        self.0.configure_decode(e, o, g)
    }
    fn configure_attention(&mut self, s: u32) -> Result<()> {
        self.0.configure_attention(s)
    }
    fn materialize_transformer_window(&mut self, f: u32, c: u32) -> Result<()> {
        self.0.materialize_transformer_window(f, c)
    }
    fn finish(&mut self, o: MemoryRunOutcome) -> Result<()> {
        self.0.finish(o)
    }
}
struct Provider {
    descriptor: ModelDescriptor,
    calls: Shared,
    mode: Mode,
}
impl Generator for Provider {
    fn descriptor(&self) -> &ModelDescriptor {
        &self.descriptor
    }
    fn validate(&self, _: &GenerationRequest) -> Result<()> {
        Ok(())
    }
    fn begin_memory_strategy_request(
        &self,
        _: &MemoryRunContext,
    ) -> Result<Option<Box<dyn MemoryRequestScope + '_>>> {
        self.calls.borrow_mut().begin += 1;
        let scope = Scope {
            calls: self.calls.clone(),
            mode: self.mode,
        };
        Ok(Some(if matches!(self.mode, Mode::LegacyOnly) {
            Box::new(LegacyScope(scope))
        } else {
            Box::new(scope)
        }))
    }
    fn generate(
        &self,
        request: &GenerationRequest,
        _: &mut dyn FnMut(Progress),
    ) -> Result<GenerationOutput> {
        let mut calls = self.calls.borrow_mut();
        calls.body += 1;
        calls.selected_in_body = mlx_gen::ios_quantum::native_retirement_binding().is_some();
        assert_eq!(
            calls.request_address, request as *const GenerationRequest as usize,
            "actual delegate must see original immutable request"
        );
        match self.mode {
            Mode::BodyError => Err(Error::Unsupported("actual body error".into())),
            Mode::Canceled => Err(Error::Canceled),
            _ => Ok(GenerationOutput::Images(vec![])),
        }
    }
}
fn bound(
    mode: Mode,
    required: bool,
) -> (BoundGenerator, GenerationRequest, Shared, tempfile::TempDir) {
    use crate::sdxl::tests::{admit_fixture_rows, budget, candidate, fixture, DEVICE};
    let registry = crate::provider_registry().unwrap();
    let root = fixture();
    let mut candidate = candidate(&registry, root.path(), "sdxl", 512, None, "image_to_image");
    if let IosWork::Generate(request) = &mut candidate.work {
        if let Some(Conditioning::Reference { image, .. }) = request.conditioning.first_mut() {
            *image = Image {
                width: 1024,
                height: 1024,
                pixels: vec![17; 3 * 1024 * 1024],
            };
        }
    }
    let row = crate::sdxl::tests::capture(
        &registry,
        &candidate,
        crate::evidence::ANCHOR_MARGIN_FLOOR_BYTES + (1 << 20),
    );
    let mut receipt = admit_fixture_rows(
        &registry,
        std::slice::from_ref(&candidate),
        DEVICE,
        budget(),
        &[row],
    )
    .unwrap();
    receipt.native_retirement = if required {
        mlx_gen::ios_pricing::IosNativeRetirementExecution::ProviderLifecycle
    } else {
        mlx_gen::ios_pricing::IosNativeRetirementExecution::Ordinary
    };
    let request = match &candidate.work {
        IosWork::Generate(request) => request.clone(),
        _ => unreachable!(),
    };
    let descriptor = registry
        .generators()
        .map(|r| (r.descriptor)())
        .find(|d| d.id == crate::sdxl::SDXL_ID)
        .unwrap();
    let calls = Rc::new(RefCell::new(Calls::default()));
    let delegate = calls.clone();
    let generator = receipt
        .load_generator_using(
            |_| {
                Ok(Box::new(Provider {
                    descriptor,
                    calls: delegate,
                    mode,
                }))
            },
            Box::new(|_| Ok(())),
        )
        .unwrap();
    (generator, request, calls, root)
}
#[test]
fn required_generation_establishes_real_scope_without_optional_begin_or_media_clone() {
    cpu_test(
        "required_generation_establishes_real_scope_without_optional_begin_or_media_clone",
        || {
            let (generator, request, calls, _root) = bound(Mode::Good, true);
            let expected = match request.conditioning.first() {
                Some(Conditioning::Reference { image, .. }) => image.pixels.as_ptr() as usize,
                _ => unreachable!(),
            };
            let (result, allocated) = crate::binding::tests::allocations_during(|| {
                generator.generate(&request, &mut |_| {})
            });
            result.unwrap();
            let calls = calls.borrow();
            assert_eq!(
                (calls.begin, calls.mutable, calls.readonly, calls.body),
                (1, 0, 1, 1)
            );
            assert_eq!(calls.pixels_address, expected);
            assert!(
                allocated < 1024 * 1024,
                "lifecycle cloned pixel-sized media: {allocated}B"
            );
            assert!(calls.selected_in_body);
            assert_eq!(calls.finish, vec![MemoryRunOutcome::Complete]);
            assert_eq!(mlx_gen::ios_quantum::native_retirement_binding(), None);
        },
    );
}
#[test]
fn refused_or_default_unsupported_configuration_stops_before_body() {
    cpu_test(
        "refused_or_default_unsupported_configuration_stops_before_body",
        || {
            for mode in [Mode::RefuseConfiguration, Mode::LegacyOnly] {
                let (generator, request, calls, _root) = bound(mode, true);
                assert!(generator.generate(&request, &mut |_| {}).is_err());
                let calls = calls.borrow();
                assert_eq!(calls.body, 0);
                assert_eq!(calls.finish.len(), 1);
                assert_eq!(mlx_gen::ios_quantum::native_retirement_binding(), None);
            }
        },
    );
}
#[test]
fn matched_external_configuration_is_preserved_and_not_implicitly_active() {
    cpu_test(
        "matched_external_configuration_is_preserved_and_not_implicitly_active",
        || {
            let (generator, mut request, calls, _root) = bound(Mode::Good, true);
            let mut scope = generator
                .begin_memory_strategy_request(generator.admitted_context())
                .unwrap()
                .unwrap();
            scope.configure_request(&mut request).unwrap();
            assert_eq!(
                mlx_gen::ios_quantum::native_retirement_binding(),
                None,
                "holding a configured scope alone must not change unrelated scheduling"
            );
            calls.borrow_mut().request_address = &request as *const GenerationRequest as usize;
            generator.generate(&request, &mut |_| {}).unwrap();
            assert_eq!(
                (
                    calls.borrow().begin,
                    calls.borrow().mutable,
                    calls.borrow().readonly
                ),
                (1, 1, 0)
            );
            assert!(
                calls.borrow().finish.is_empty(),
                "executor must not finish a caller-owned external scope"
            );
            scope.finish(MemoryRunOutcome::Complete).unwrap();
        },
    );
}
#[test]
fn different_external_configuration_does_not_replace_actual_scope() {
    cpu_test(
        "different_external_configuration_does_not_replace_actual_scope",
        || {
            let (first, mut request, _, _first_root) = bound(Mode::Good, true);
            let mut external = first
                .begin_memory_strategy_request(first.admitted_context())
                .unwrap()
                .unwrap();
            external.configure_request(&mut request).unwrap();
            let (second, request, calls, _second_root) = bound(Mode::Good, true);
            second.generate(&request, &mut |_| {}).unwrap();
            assert_eq!((calls.borrow().begin, calls.borrow().readonly), (1, 1));
            external.finish(MemoryRunOutcome::Complete).unwrap();
        },
    );
}
#[test]
fn public_registration_cannot_skip_actual_provider_configuration() {
    cpu_test(
        "public_registration_cannot_skip_actual_provider_configuration",
        || {
            let (generator, request, calls, _root) = bound(Mode::Good, true);
            let _asserted = mlx_gen::ios_quantum::ConfiguredNativeRetirement::register(
                generator.retirement_binding(),
            );
            generator.generate(&request, &mut |_| {}).unwrap();
            assert_eq!((calls.borrow().begin, calls.borrow().readonly), (1, 1));
        },
    );
}
#[test]
fn unrelated_active_lifecycle_refuses_required_work_before_body() {
    cpu_test(
        "unrelated_active_lifecycle_refuses_required_work_before_body",
        || {
            let (generator, request, calls, _root) = bound(Mode::Good, true);
            let mut wrong = generator.retirement_binding();
            wrong.work[0] ^= 1;
            let registration = mlx_gen::ios_quantum::ConfiguredNativeRetirement::register(wrong);
            let _active = mlx_gen::ios_quantum::NativeRetirementScope::enter_configured(
                &registration.lease(),
            )
            .unwrap();
            assert!(generator.generate(&request, &mut |_| {}).is_err());
            assert_eq!((calls.borrow().begin, calls.borrow().body), (0, 0));
        },
    );
}
#[test]
fn ordinary_generation_does_not_inherit_nested_retirement_or_create_scope() {
    cpu_test(
        "ordinary_generation_does_not_inherit_nested_retirement_or_create_scope",
        || {
            let (generator, request, calls, _root) = bound(Mode::Good, false);
            calls.borrow_mut().request_address = &request as *const GenerationRequest as usize;
            let binding = generator.retirement_binding();
            let registration = mlx_gen::ios_quantum::ConfiguredNativeRetirement::register(binding);
            let _active = mlx_gen::ios_quantum::NativeRetirementScope::enter_configured(
                &registration.lease(),
            )
            .unwrap();
            generator.generate(&request, &mut |_| {}).unwrap();
            assert_eq!(
                (
                    calls.borrow().begin,
                    calls.borrow().readonly,
                    calls.borrow().body
                ),
                (0, 0, 1)
            );
            assert!(!calls.borrow().selected_in_body);
            assert_eq!(
                mlx_gen::ios_quantum::native_retirement_binding(),
                Some(binding)
            );
        },
    );
}
#[test]
fn owned_scope_finishes_errors_and_cancellation_preserving_primary_class() {
    cpu_test(
        "owned_scope_finishes_errors_and_cancellation_preserving_primary_class",
        || {
            for mode in [Mode::BodyError, Mode::Canceled] {
                let (generator, request, calls, _root) = bound(mode, true);
                let result = generator.generate(&request, &mut |_| {});
                match mode {
                    Mode::Canceled => assert!(matches!(result, Err(Error::Canceled))),
                    _ => assert!(
                        matches!(result,Err(Error::Unsupported(reason)) if reason=="actual body error")
                    ),
                }
                assert_eq!(calls.borrow().body, 1);
                assert_eq!(calls.borrow().finish.len(), 1);
                assert_eq!(mlx_gen::ios_quantum::native_retirement_binding(), None);
            }
        },
    );
}
