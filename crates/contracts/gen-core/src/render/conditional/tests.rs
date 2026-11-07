use super::*;
use std::{
    cell::{Cell, RefCell},
    collections::VecDeque,
    rc::Rc,
};

fn input() -> ConditionalInput {
    ConditionalInput {
        provider_id: "sensenova_u1_8b".into(),
        prompt_sha256: [7; 32],
        seed: 42,
        width: 64,
        height: 64,
        cell: 16,
        input_images: 0,
        source_images: Vec::new(),
        prefill_tokens: [10, 3, 2],
        prefill_hidden_width: 4096,
        prefill_vision_hidden_width: 1024,
        source_patch_size: 8,
        prefill_attention_heads: 32,
        max_new_tokens: 10,
        max_images: 3,
        steps_per_image: 4,
        model_resident_bytes: 1000,
        cache_token_bytes: 32,
        step_scratch_bytes: 2000,
    }
}
fn ceilings() -> ConditionalCeilings {
    ConditionalCeilings {
        max_width: 1024,
        max_height: 1024,
        max_input_images: 4,
        max_source_width: 1024,
        max_source_height: 1024,
        max_source_token_grid: 4096,
        max_source_live_bytes: 100_000_000,
        max_prefill_hidden_width: 8192,
        max_prefill_vision_hidden_width: 4096,
        max_prefill_attention_heads: 64,
        max_prefill_scratch_bytes: 10_000_000_000,
        max_token_grid: 4096,
        max_new_tokens: 128,
        max_images: 8,
        max_steps_per_image: 50,
        max_cache_tokens_per_cache: 10000,
        max_peak_bytes: 100_000_000,
        max_total_work: 1_000_000_000,
    }
}
fn plan() -> ConditionalPlan {
    ConditionalPlan::compile(input(), ceilings()).unwrap()
}
fn segment(emitted: u32, stop: RolloutStop) -> TextRollout {
    TextRollout {
        text: "t".repeat(emitted as usize),
        emitted,
        stop,
    }
}
#[derive(Default)]
struct Probe {
    calls: Rc<RefCell<Vec<String>>>,
    drops: Rc<Cell<usize>>,
}
struct Cache(Rc<Cell<usize>>);
impl Drop for Cache {
    fn drop(&mut self) {
        self.0.set(self.0.get() + 1);
    }
}
struct Mock {
    probe: Probe,
    script: VecDeque<TextRollout>,
    cancel: CancelFlag,
    cancel_at: Option<&'static str>,
    source_images: Vec<SourceImageGeometry>,
    prefill_hidden_width: u32,
    prefill_tokens: [u32; 3],
}
impl Mock {
    fn new(script: Vec<TextRollout>) -> Self {
        Self {
            probe: Probe::default(),
            script: script.into(),
            cancel: CancelFlag::new(),
            cancel_at: None,
            source_images: Vec::new(),
            prefill_hidden_width: 4096,
            prefill_tokens: [10, 3, 2],
        }
    }
    fn log(&self, call: String) {
        self.probe.calls.borrow_mut().push(call.clone());
        if self.cancel_at == Some(call.as_str()) {
            self.cancel.cancel();
        }
    }
    fn calls(&self) -> Vec<String> {
        self.probe.calls.borrow().clone()
    }
}
impl ConditionalExecutor for Mock {
    type Cache = Cache;
    type Context = ();
    type Image = u32;
    fn source_images(&self) -> Result<Vec<SourceImageGeometry>> {
        Ok(self.source_images.clone())
    }
    fn prefill_model_shape(&self) -> Result<PrefillModelShape> {
        Ok(PrefillModelShape {
            hidden_width: self.prefill_hidden_width,
            vision_hidden_width: 1024,
            source_patch_size: 8,
            attention_heads: 32,
        })
    }
    fn prefill_token_counts(&self) -> Result<[u32; 3]> {
        Ok(self.prefill_tokens)
    }
    fn prefill(&mut self, _: &ConditionalPlan) -> Result<((), [Cache; 3])> {
        self.log("prefill".into());
        Ok(((), std::array::from_fn(|_| Cache(self.probe.drops.clone()))))
    }
    fn rollout(&mut self, _: &mut (), _: &mut [Cache; 3], granted: u32) -> Result<TextRollout> {
        self.log(format!("rollout:{granted}"));
        self.script
            .pop_front()
            .ok_or_else(|| refused("unscripted rollout"))
    }
    fn denoise_step(
        &mut self,
        _: &mut (),
        _: &mut [Cache; 3],
        image: u32,
        step: u32,
    ) -> Result<()> {
        self.log(format!("denoise:{image}:{step}"));
        Ok(())
    }
    fn finish_image(&mut self, _: &mut (), image: u32) -> Result<u32> {
        self.log(format!("finish:{image}"));
        Ok(image)
    }
    fn feedback(&mut self, _: &mut (), _: &mut [Cache; 3], image: u32, _: &u32) -> Result<()> {
        self.log(format!("feedback:{image}"));
        Ok(())
    }
}
fn run(mock: &mut Mock, plan: &ConditionalPlan) -> (Result<ConditionalOutput<u32>>, Vec<Progress>) {
    let mut events = Vec::new();
    let result = run_conditional(plan, plan.input(), &mock.cancel.clone(), mock, &mut |p| {
        events.push(p)
    });
    (result, events)
}
#[test]
fn two_images_then_end_preserves_order_and_fills_progress_without_inventing_steps() {
    let p = plan();
    let mut mock = Mock::new(vec![
        segment(1, RolloutStop::Image),
        segment(1, RolloutStop::Image),
        segment(0, RolloutStop::End),
    ]);
    let (result, events) = run(&mut mock, &p);
    let output = result.unwrap();
    assert_eq!(output.images, [0, 1]);
    assert_eq!(output.receipt.actual_outer_steps, 8);
    assert_eq!(output.receipt.reported_final_steps, Some(12));
    assert_eq!(output.receipt.feedback_calls, 2);
    assert_eq!(output.receipt.text_draws, 5);
    assert_eq!(output.receipt.terminal, TerminalReason::EndToken);
    assert_eq!(
        events
            .iter()
            .filter(|p| matches!(p, Progress::Step { .. }))
            .count(),
        9
    );
    assert!(events.contains(&Progress::Step {
        current: 12,
        total: 12
    }));
    assert_eq!(mock.probe.drops.get(), 3);
    assert_eq!(
        mock.calls()
            .iter()
            .filter(|s| s.starts_with("rollout"))
            .count(),
        3
    );
    assert_eq!(
        mock.calls(),
        [
            "prefill",
            "rollout:10",
            "denoise:0:0",
            "denoise:0:1",
            "denoise:0:2",
            "denoise:0:3",
            "finish:0",
            "feedback:0",
            "rollout:8",
            "denoise:1:0",
            "denoise:1:1",
            "denoise:1:2",
            "denoise:1:3",
            "finish:1",
            "feedback:1",
            "rollout:6",
        ]
    );
    assert!(matches!(
        output.receipt.stages.last(),
        Some(RealizedStage::Terminal { .. })
    ));
}
#[test]
fn zero_image_end_has_no_image_bar_or_cache_leak() {
    let p = plan();
    let mut mock = Mock::new(vec![segment(0, RolloutStop::End)]);
    let (result, events) = run(&mut mock, &p);
    let output = result.unwrap();
    assert_eq!(output.receipt.images, 0);
    assert_eq!(output.receipt.actual_outer_steps, 0);
    assert_eq!(output.receipt.reported_final_steps, None);
    assert!(
        !events
            .iter()
            .any(|p| matches!(p, Progress::Step { .. } | Progress::Decoding))
    );
    assert_eq!(mock.probe.drops.get(), 3);
}
#[test]
fn exhausted_budget_after_feedback_never_forces_another_rollout() {
    let mut i = input();
    i.max_new_tokens = 2;
    let p = ConditionalPlan::compile(i, ceilings()).unwrap();
    let mut mock = Mock::new(vec![segment(1, RolloutStop::Image)]);
    let (result, _) = run(&mut mock, &p);
    let output = result.unwrap();
    assert_eq!(output.receipt.terminal, TerminalReason::TokenLimit);
    assert_eq!(output.receipt.text_draws, 2);
    assert_eq!(output.receipt.images, 1);
    assert_eq!(
        mock.calls()
            .iter()
            .filter(|s| s.starts_with("rollout"))
            .count(),
        1
    );
    assert_eq!(mock.probe.drops.get(), 3);
}
#[test]
fn image_limit_prevents_extra_denoise() {
    let mut i = input();
    i.max_images = 1;
    let p = ConditionalPlan::compile(i, ceilings()).unwrap();
    let mut mock = Mock::new(vec![
        segment(0, RolloutStop::Image),
        segment(0, RolloutStop::Image),
    ]);
    let (result, _) = run(&mut mock, &p);
    let output = result.unwrap();
    assert_eq!(output.receipt.terminal, TerminalReason::ImageLimit);
    assert_eq!(output.receipt.images, 1);
    assert_eq!(
        mock.calls()
            .iter()
            .filter(|s| s.starts_with("denoise"))
            .count(),
        4
    );
}
#[test]
fn cancellation_in_second_denoise_stops_output_feedback_and_releases_caches() {
    let p = plan();
    let mut mock = Mock::new(vec![
        segment(0, RolloutStop::Image),
        segment(0, RolloutStop::Image),
    ]);
    mock.cancel_at = Some("denoise:1:1");
    let (result, events) = run(&mut mock, &p);
    assert!(matches!(result, Err(Error::Canceled)));
    assert_eq!(mock.calls().last().unwrap(), "denoise:1:1");
    assert!(!mock.calls().contains(&"feedback:1".into()));
    assert_eq!(mock.probe.drops.get(), 3);
    assert_eq!(
        events
            .iter()
            .filter(|p| matches!(p, Progress::Step { .. }))
            .count(),
        5
    );
}
#[test]
fn cancellation_in_second_feedback_stops_publication_and_releases_caches() {
    let p = plan();
    let mut mock = Mock::new(vec![
        segment(0, RolloutStop::Image),
        segment(0, RolloutStop::Image),
    ]);
    mock.cancel_at = Some("feedback:1");
    let (result, events) = run(&mut mock, &p);
    assert!(matches!(result, Err(Error::Canceled)));
    assert_eq!(mock.calls().last().unwrap(), "feedback:1");
    assert_eq!(mock.probe.drops.get(), 3);
    assert!(!events.contains(&Progress::Decoding));
}
#[test]
fn guard_rejects_reuse_and_overdraw() {
    let p = plan();
    let mut guard = ConditionalGuard::new(&p).unwrap();
    guard.prefilled().unwrap();
    assert!(guard.prefilled().is_err());
    let grant = guard.begin_rollout().unwrap();
    assert_eq!(grant, 10);
    assert!(guard.begin_rollout().is_err());
    assert!(
        guard
            .finish_rollout(&segment(10, RolloutStop::End))
            .is_err()
    );
    assert!(
        guard
            .finish_rollout(&segment(0, RolloutStop::Image))
            .is_ok()
    );
    assert!(guard.begin_rollout().is_err());
    assert!(guard.check_step(1, 0).is_err());
}
#[test]
fn admission_rejects_caps_overflow_and_forged_plan() {
    let mut i = input();
    i.width = 65;
    assert!(ConditionalPlan::compile(i, ceilings()).is_err());
    let mut i = input();
    i.model_resident_bytes = u64::MAX;
    assert!(ConditionalPlan::compile(i, ceilings()).is_err());
    let mut c = ceilings();
    c.max_total_work = 1;
    assert!(ConditionalPlan::compile(input(), c).is_err());
    let p = plan();
    let serialized = serde_json::to_string(&p).unwrap();
    let decoded: ConditionalPlan = serde_json::from_str(&serialized).unwrap();
    decoded.validate().unwrap();
    let mut forged = serde_json::to_value(&p).unwrap();
    forged["budget"]["max_outer_steps"] = serde_json::json!(99);
    let forged: ConditionalPlan = serde_json::from_value(forged).unwrap();
    assert!(forged.validate().is_err());
    assert!(
        p.check_input(&ConditionalInput { seed: 3, ..input() })
            .is_err()
    );
}

#[test]
fn source_images_require_real_expanded_prefixes_and_live_storage() {
    let source = SourceImageGeometry {
        width: 64,
        height: 64,
        storage_bytes: 64 * 64 * 3 * 4,
    };
    let mut i = input();
    i.input_images = 4;
    i.source_images = vec![source.clone(); 4];
    i.prefill_tokens = [1, 1, 1];
    assert!(ConditionalPlan::compile(i.clone(), ceilings()).is_err());

    // Four 16-token grids each add an <img> and </img> marker to both prefixes.
    i.prefill_tokens = [73, 73, 1];
    let p = ConditionalPlan::compile(i.clone(), ceilings()).unwrap();
    assert_eq!(p.budget().source_image_tokens, 64);
    assert_eq!(p.budget().source_live_bytes, 4 * 6 * 64 * 64 * 3 * 4);
    assert!(p.budget().peak_live_bytes > plan().budget().peak_live_bytes);
    assert!(p.budget().total_work_units > plan().budget().total_work_units);

    let mut c = ceilings();
    c.max_source_live_bytes = p.budget().source_live_bytes - 1;
    assert!(ConditionalPlan::compile(i.clone(), c).is_err());
    let mut bad = i.clone();
    bad.source_images[0].storage_bytes = 1;
    assert!(ConditionalPlan::compile(bad, ceilings()).is_err());
    let mut bad = i.clone();
    bad.source_images[0].width = 65;
    assert!(ConditionalPlan::compile(bad, ceilings()).is_err());
    let mut bad = i.clone();
    bad.source_images.pop();
    assert!(ConditionalPlan::compile(bad, ceilings()).is_err());

    let mut mock = Mock::new(vec![segment(0, RolloutStop::End)]);
    let (result, _) = run(&mut mock, &p);
    assert!(result.is_err());
    assert!(mock.calls().is_empty()); // mismatch is refused before any provider callback
    mock.source_images = vec![source; 4];
    let (result, _) = run(&mut mock, &p);
    assert!(result.is_err());
    assert!(mock.calls().is_empty()); // tokenized prefix lengths are checked too
    mock.prefill_tokens = i.prefill_tokens;
    let (result, _) = run(&mut mock, &p);
    assert!(result.is_ok());
}

#[test]
fn dense_four_large_source_prefill_is_priced_without_allocating_it() {
    let source = SourceImageGeometry {
        width: 1024,
        height: 1024,
        storage_bytes: 1024 * 1024 * 3 * 4,
    };
    let mut i = input();
    i.input_images = 4;
    i.source_images = vec![source; 4];
    i.prefill_tokens = [16_393, 16_393, 1];
    let mut c = ceilings();
    c.max_source_live_bytes = 1_000_000_000;
    c.max_cache_tokens_per_cache = 100_000;
    c.max_peak_bytes = 100_000_000_000;
    c.max_total_work = 100_000_000_000;
    c.max_prefill_scratch_bytes = 3_000_000_000;
    assert!(ConditionalPlan::compile(i.clone(), c.clone()).is_err());
    c.max_prefill_scratch_bytes = 100_000_000_000;
    let p = ConditionalPlan::compile(i, c.clone()).unwrap();
    assert_eq!(p.budget().source_image_tokens, 16_384);
    assert!(p.budget().prefill_scratch_bytes > 3_000_000_000);
    assert!(p.budget().peak_live_bytes > p.budget().prefill_scratch_bytes);
    c.max_peak_bytes = p.budget().peak_live_bytes - 1;
    assert!(ConditionalPlan::compile(p.input().clone(), c).is_err());

    let mut mock = Mock::new(vec![segment(0, RolloutStop::End)]);
    mock.source_images = p.input().source_images.clone();
    mock.prefill_hidden_width = 2048;
    let (result, _) = run(&mut mock, &p);
    assert!(result.is_err());
    assert!(mock.calls().is_empty());
}
