//! Resource facts bind full ordered reference geometry, not just output/count.
use gen_core::flux3::references::ReferenceOutputSize;
use gen_core::{
    Conditioning, GenerationRequest, Image, MemoryGeometry, MemoryPhase, MemoryRequestScope,
    MemoryRunOutcome,
};
use gen_core_flux3::references;
use std::{cell::Cell, rc::Rc};

struct Probe(Rc<Cell<usize>>);
impl MemoryRequestScope for Probe {
    fn configure_request(&mut self, _: &mut GenerationRequest) -> gen_core::Result<()> {
        self.0.set(self.0.get() + 1);
        Ok(())
    }
    fn enter_phase(&mut self, _: MemoryPhase) -> gen_core::Result<()> {
        Ok(())
    }
    fn leave_phase(&mut self, _: MemoryPhase) -> gen_core::Result<()> {
        Ok(())
    }
    fn configure_decode(&mut self, _: u32, _: u32, _: MemoryGeometry) -> gen_core::Result<()> {
        Ok(())
    }
    fn configure_attention(&mut self, _: u32) -> gen_core::Result<()> {
        Ok(())
    }
    fn materialize_transformer_window(&mut self, _: u32, _: u32) -> gen_core::Result<()> {
        Ok(())
    }
    fn finish(&mut self, _: MemoryRunOutcome) -> gen_core::Result<()> {
        Ok(())
    }
}
fn request() -> GenerationRequest {
    GenerationRequest {
        conditioning: [(128, 64), (64, 128)]
            .into_iter()
            .map(|(width, height)| Conditioning::Reference {
                image: Image {
                    width,
                    height,
                    pixels: vec![37; (width * height * 3) as usize],
                },
                strength: None,
            })
            .collect(),
        flux3_reference_output: Some(ReferenceOutputSize {
            width: Some(128),
            height: Some(128),
        }),
        steps: Some(7),
        guidance: Some(1.),
        seed: Some(17),
        count: 2,
        ..Default::default()
    }
}
fn geometry() -> MemoryGeometry {
    MemoryGeometry {
        width: 128,
        height: 128,
        batch: 2,
        frames: 1,
        reference_count: 2,
    }
}

#[test]
fn facts_keep_every_non_square_canvas_and_reference_only_tokens() {
    let req = request();
    let plan = references::request_plan(&req).unwrap().unwrap();
    let facts = plan.memory_resources().unwrap();
    assert_eq!(facts.prepared_canvases, vec![(128, 64), (64, 128)]);
    assert_eq!(facts.conditioning_tokens, 64); // 32 + 32, no CFG/text/job multiplier.
    let mut corrupt = plan.clone();
    corrupt.conditioning_tokens = 32;
    assert!(corrupt.memory_resources().is_err());
    let mut corrupt = plan;
    corrupt.canvases.pop();
    assert!(corrupt.memory_resources().is_err());
}

#[test]
fn full_guard_rejects_forged_tokens_shapes_order_and_missing_facts_before_delegate() {
    let original = request();
    let facts = references::request_plan(&original)
        .unwrap()
        .unwrap()
        .memory_resources()
        .unwrap();
    // Each invalid variant keeps output resolution and reference COUNT unchanged.
    let mut bad_tokens = facts.clone();
    bad_tokens.conditioning_tokens = 63;
    let mut variants = vec![Some(bad_tokens)];
    let mut reordered = facts.clone();
    reordered.prepared_canvases.reverse();
    variants.push(Some(reordered));
    let mut reshaped = facts.clone();
    reshaped.prepared_canvases[0] = (64, 128);
    variants.push(Some(reshaped));
    variants.push(None);
    for bad in variants {
        let seen = Rc::new(Cell::new(0));
        let mut scope = references::guard_memory_scope_with_resources(
            Box::new(Probe(seen.clone())),
            geometry(),
            bad,
        );
        assert!(scope.configure_request(&mut original.clone()).is_err());
        assert_eq!(seen.get(), 0);
    }
    let seen = Rc::new(Cell::new(0));
    let mut scope = references::guard_memory_scope_with_resources(
        Box::new(Probe(seen.clone())),
        geometry(),
        Some(facts),
    );
    let mut accepted = original.clone();
    scope.configure_request(&mut accepted).unwrap();
    assert_eq!(seen.get(), 1);
    assert_eq!((accepted.width, accepted.height), (128, 128));
    assert_eq!(accepted.steps, original.steps);
    assert_eq!(accepted.guidance, original.guidance);
    assert_eq!(accepted.seed, original.seed);
    assert_eq!(accepted.count, original.count);
    assert_eq!(
        accepted.flux3_reference_output,
        original.flux3_reference_output
    );
    assert_eq!(accepted.conditioning.len(), 2);
}

#[test]
fn legacy_guard_and_standalone_auto_intent_remain_usable() {
    let seen = Rc::new(Cell::new(0));
    let mut legacy = references::guard_memory_scope(Box::new(Probe(seen.clone())), geometry());
    legacy.configure_request(&mut request()).unwrap();
    let mut standalone = GenerationRequest {
        conditioning: vec![Conditioning::Reference {
            image: Image {
                width: 2032,
                height: 64,
                pixels: vec![9; 2032 * 64 * 3],
            },
            strength: None,
        }],
        flux3_reference_output: Some(ReferenceOutputSize::default()),
        ..Default::default()
    };
    let plan = references::normalize_request(&mut standalone)
        .unwrap()
        .unwrap();
    assert_eq!((standalone.width, standalone.height), (2032, 64));
    assert_eq!(
        plan.memory_resources().unwrap().prepared_canvases,
        vec![(2032, 64)]
    );
    assert_eq!(
        standalone.flux3_reference_output,
        Some(ReferenceOutputSize::default())
    );
    let mut none = GenerationRequest::default();
    let mut scope = references::guard_memory_scope_with_resources(
        Box::new(Probe(seen.clone())),
        MemoryGeometry {
            width: 1024,
            height: 1024,
            batch: 1,
            frames: 1,
            reference_count: 0,
        },
        None,
    );
    scope.configure_request(&mut none).unwrap();
    assert_eq!(seen.get(), 2);
}
