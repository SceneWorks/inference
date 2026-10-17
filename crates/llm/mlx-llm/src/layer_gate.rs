//! An optional per-layer pre-construction gate for the decoder stack (epic 16774).
//!
//! A media consumer that reuses this engine as a text tower (Iris's Qwen3-VL encoder) on a
//! memory-capped device installs a gate that checks live OS headroom BEFORE each decoder layer's
//! graph is built, and refuses with a typed error. With no gate installed — every macOS, CUDA, CPU
//! and LLM path — the stack runs exactly as before: one `None` check per layer.

use std::cell::RefCell;

type Gate = Box<dyn FnMut(usize) -> std::result::Result<(), String>>;

thread_local! {
    static GATE: RefCell<Option<Gate>> = const { RefCell::new(None) };
}

/// Run `body` with `gate(layer_index)` called before each decoder layer is constructed on this
/// thread. A gate error stops the stack before that layer, as [`crate::Error::Unsupported`]. The
/// previous gate (if any) is restored afterwards, including on unwind.
pub fn with_layer_gate<T>(
    gate: impl FnMut(usize) -> std::result::Result<(), String> + 'static,
    body: impl FnOnce() -> T,
) -> T {
    struct Restore(Option<Gate>);
    impl Drop for Restore {
        fn drop(&mut self) {
            let previous = self.0.take();
            GATE.with(|slot| *slot.borrow_mut() = previous);
        }
    }
    let previous = GATE.with(|slot| slot.borrow_mut().replace(Box::new(gate)));
    let _restore = Restore(previous);
    body()
}

/// Consult the installed gate (if any) before constructing decoder layer `index`.
pub(crate) fn before_layer(index: usize) -> crate::Result<()> {
    let Some(mut gate) = GATE.with(|slot| slot.borrow_mut().take()) else {
        return Ok(());
    };
    let result = gate(index);
    GATE.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_none() {
            *slot = Some(gate);
        }
    });
    result.map_err(crate::Error::Unsupported)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::rc::Rc;

    #[test]
    fn no_gate_is_a_pass_through_and_a_gate_refuses_before_its_layer() {
        assert!(before_layer(0).is_ok());
        let seen = Rc::new(Cell::new(0usize));
        let sink = seen.clone();
        let refused = with_layer_gate(
            move |i| {
                sink.set(sink.get() + 1);
                if i == 2 {
                    Err("layer 2 refused".into())
                } else {
                    Ok(())
                }
            },
            || (0..5).map(before_layer).position(|r| r.is_err()),
        );
        assert_eq!(refused, Some(2));
        assert_eq!(
            seen.get(),
            3,
            "the gate stops the walk at the refused layer"
        );
        assert!(
            before_layer(0).is_ok(),
            "the gate is removed after its body"
        );
    }
}
