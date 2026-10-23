use super::*;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

struct OutputAllocator;
thread_local! {
    static ACTIVE: Cell<bool> = const { Cell::new(false) };
    static CREDIT: Cell<usize> = const { Cell::new(0) };
    static UNBILLED: Cell<usize> = const { Cell::new(0) };
    static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
}
fn allocate(bytes: usize) {
    if ACTIVE.try_with(Cell::get).unwrap_or(false) {
        ALLOCATIONS.with(|value| value.set(value.get() + 1));
        CREDIT.with(|credit| {
            if bytes > credit.get() {
                UNBILLED.with(|value| value.set(value.get() + 1));
            }
            credit.set(credit.get().saturating_sub(bytes));
        });
    }
}
unsafe impl GlobalAlloc for OutputAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        allocate(layout.size());
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        allocate(layout.size());
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        allocate(size);
        unsafe { System.realloc(pointer, layout, size) }
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }
}
#[global_allocator]
static ALLOCATOR: OutputAllocator = OutputAllocator;

fn output(row: &Value, refused: bool) -> (Result<String, ()>, usize, usize, usize) {
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            ACTIVE.with(|value| value.set(false));
        }
    }
    let mut calls = 0;
    let mut billed = 0;
    let mut next = |bytes| {
        calls += 1;
        billed = bytes as usize;
        CREDIT.with(|value| value.set(0));
        if refused {
            return Err(mlx_gen::Error::Canceled);
        }
        CREDIT.with(|value| value.set(bytes as usize));
        Ok(())
    };
    CREDIT.with(|value| value.set(0));
    UNBILLED.with(|value| value.set(0));
    ALLOCATIONS.with(|value| value.set(0));
    ACTIVE.with(|value| value.set(true));
    let reset = Reset;
    let result =
        serialize_host_diagnostic_with_capacity("SMOKE: FAIL\nCAMPAIGN_JSON ", row, &mut next);
    drop(reset);
    assert_eq!(
        UNBILLED.with(Cell::get),
        0,
        "counter and serializer must allocate nothing before capacity"
    );
    (result, billed, calls, ALLOCATIONS.with(Cell::get))
}

#[test]
fn host_diagnostic_counts_actual_overlay_and_allocates_output_once_after_capacity() {
    let escaped = "\0\n\"\\é".repeat(100_000);
    for row in [
        json!({}),
        json!({"status":"diagnostic_failure", "case":"unexecuted", "production_success":false, "reason":escaped}),
        json!({"evidence_key":{"old":"must not clone/retain"},"reason":"host failure"}),
        json!({"evidence_key":null,"reason":"host failure"}),
    ] {
        let original = row.clone();
        let (result, billed, calls, allocations) = output(&row, false);
        let text = result.unwrap();
        assert_eq!(calls, 1);
        assert_eq!(
            allocations, 1,
            "the actual emitted String is the only new owner"
        );
        assert_eq!(
            text.len(),
            billed,
            "actual output length, not a fabricated expected count"
        );
        assert!(text.starts_with("SMOKE: FAIL\nCAMPAIGN_JSON "));
        let parsed: Value = serde_json::from_str(
            text.lines()
                .nth(1)
                .unwrap()
                .strip_prefix("CAMPAIGN_JSON ")
                .unwrap(),
        )
        .unwrap();
        assert!(parsed["evidence_key"].is_null());
        assert_eq!(
            row, original,
            "the count pass must borrow, not mutate/copy the JSON"
        );
    }
}

#[test]
fn host_diagnostic_capacity_refusal_and_invalid_row_allocate_no_output_or_error_string() {
    let row = json!({"status":"diagnostic_failure", "production_success":false, "reason":"real iOS device required"});
    let (result, _, calls, allocations) = output(&row, true);
    assert!(result.is_err());
    assert_eq!(calls, 1);
    assert_eq!(allocations, 0);
    let (result, _, calls, allocations) = output(&Value::Null, false);
    assert!(result.is_err());
    assert_eq!(calls, 0);
    assert_eq!(allocations, 0);
}
