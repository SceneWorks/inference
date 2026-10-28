//! The JSON-constraint adapter for [`LlamaProvider`](super::LlamaProvider)'s decode (sc-27302): a
//! `core_llm::JsonConstraint` behind the engine's constraint-mask seams.

use super::*;

/// Adapts a `core_llm::JsonConstraint` to the engine's [`ConstraintMask`] decode seam.
pub(super) struct JsonMask<'a> {
    inner: JsonConstraint<'a>,
    table: &'a ConstraintDecodeTable,
    stop_ids: Vec<u32>,
    accepted: Vec<i32>,
    initial_reasoning: bool,
    reasoning: bool,
    reasoning_tail: String,
    allow: Vec<bool>,
}

impl<'a> JsonMask<'a> {
    pub(super) fn new(
        table: &'a ConstraintDecodeTable,
        stop_ids: impl IntoIterator<Item = u32>,
        reasoning: bool,
    ) -> Self {
        let stop_ids = stop_ids.into_iter().collect::<Vec<_>>();
        Self {
            inner: JsonConstraint::new(table, stop_ids.iter().copied()),
            table,
            stop_ids,
            accepted: Vec::new(),
            initial_reasoning: reasoning,
            reasoning,
            reasoning_tail: String::new(),
            allow: vec![false; table.pieces.len()],
        }
    }

    pub(super) fn accept_reasoning_piece(&mut self, piece: &str) {
        self.reasoning_tail.push_str(piece);
        if let Some(end) = self.reasoning_tail.find("</think>") {
            let answer = self.reasoning_tail[end + "</think>".len()..].to_string();
            self.reasoning = false;
            self.reasoning_tail.clear();
            let accepted = self.inner.accept_text(&answer);
            debug_assert!(accepted);
            return;
        }
        let keep = (1.."</think>".len())
            .rev()
            .find(|&n| self.reasoning_tail.ends_with(&"</think>"[..n]))
            .unwrap_or(0);
        if self.reasoning_tail.len() > keep {
            self.reasoning_tail
                .drain(..self.reasoning_tail.len() - keep);
        }
    }
}

impl ConstraintMask for JsonMask<'_> {
    fn allowed(&mut self) -> &[bool] {
        if !self.reasoning {
            return self.inner.allowed();
        }
        for (id, slot) in self.allow.iter_mut().enumerate() {
            if self.stop_ids.contains(&(id as u32)) {
                *slot = false;
                continue;
            }
            let mut candidate = self.reasoning_tail.clone();
            candidate.push_str(&self.table.pieces[id]);
            *slot = candidate
                .find("</think>")
                .is_none_or(|end| self.inner.allows_text(&candidate[end + "</think>".len()..]));
        }
        &self.allow
    }
    fn accept(&mut self, token: i32) {
        if self.reasoning {
            let piece = self
                .table
                .pieces
                .get(token as usize)
                .cloned()
                .unwrap_or_default();
            self.accept_reasoning_piece(&piece);
        } else {
            self.inner.accept(token as u32);
        }
        self.accepted.push(token);
    }
}

impl RewindableConstraintMask for JsonMask<'_> {
    fn checkpoint(&self) -> usize {
        self.accepted.len()
    }

    fn rewind(&mut self, checkpoint: usize) {
        self.accepted.truncate(checkpoint);
        self.inner = JsonConstraint::new(self.table, self.stop_ids.iter().copied());
        self.reasoning_tail.clear();
        self.reasoning = self.initial_reasoning;
        // Replay committed tokens to restore both the reasoning phase and JSON grammar.
        let accepted = self.accepted.clone();
        for token in accepted {
            self.accept(token);
        }
        self.accepted.truncate(checkpoint);
    }
}
