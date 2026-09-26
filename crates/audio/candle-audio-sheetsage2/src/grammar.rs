//! The decoding grammar (port of `generation_sheetsage2.PromptGrammarState` at `4f89269`).
//!
//! Greedy decoding picks the argmax **inside** the set this state allows, so the grammar is part of
//! the model's output, not a filter on it: a mask that differs from upstream's by one id changes the
//! transcription.

use crate::tokenizer::{Range, TokenType, Tokenizer, EOS};
use crate::Error;

/// Which part of an event is still incomplete.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Incomplete {
    RhythmAfterMeter,
    MelodyAfterPitch,
}

/// Grammar state of one sequence.
#[derive(Clone, Debug)]
pub struct GrammarState {
    generated_events: u64,
    in_shift: bool,
    shift_run: u32,
    payload_count: u32,
    /// `-1` = none, else the `FIELD_TO_INDEX` of the last field.
    last_field: i32,
    incomplete: Option<Incomplete>,
}

impl Default for GrammarState {
    fn default() -> Self {
        Self::new()
    }
}

impl GrammarState {
    /// The state after `<|out|>`.
    pub fn new() -> Self {
        Self {
            generated_events: 0,
            in_shift: true,
            shift_run: 0,
            payload_count: 0,
            last_field: -1,
            incomplete: None,
        }
    }

    /// Completed events so far.
    pub fn generated_events(&self) -> u64 {
        self.generated_events
    }

    /// The allowed token ranges before the next token (upstream `allowed`), as half-open runs in
    /// ascending order.
    pub fn allowed(&self, tokenizer: &Tokenizer) -> Vec<Range> {
        let mut runs = Vec::new();
        if self.payload_count > 0 {
            runs.push(Range {
                start: EOS,
                end: EOS + 1,
            });
        }
        if (self.payload_count > 0 || self.in_shift) && self.shift_run < 4 {
            runs.push(tokenizer.subbeat_shift);
        }
        match self.incomplete {
            Some(Incomplete::RhythmAfterMeter) => runs.push(tokenizer.eighth_position),
            Some(Incomplete::MelodyAfterPitch) => {
                runs.push(tokenizer.duration);
                runs.push(tokenizer.pitch);
            }
            None => {
                let last = self.last_field;
                if last < 0 {
                    runs.push(tokenizer.time);
                }
                if last < 1 {
                    runs.push(tokenizer.meter);
                    runs.push(tokenizer.eighth_position);
                }
                if last < 2 {
                    runs.push(tokenizer.structure);
                }
                if last < 3 {
                    runs.push(tokenizer.key);
                }
                if last < 4 {
                    runs.push(tokenizer.full_chord);
                }
                if last <= 5 {
                    runs.push(tokenizer.pitch);
                }
            }
        }
        runs.sort_by_key(|r| r.start);
        // Merge adjacent runs so the representation is canonical.
        let mut merged: Vec<Range> = Vec::with_capacity(runs.len());
        for run in runs {
            match merged.last_mut() {
                Some(last) if last.end >= run.start => last.end = last.end.max(run.end),
                _ => merged.push(run),
            }
        }
        merged
    }

    /// Advance past `token`; `Ok(true)` when the sequence is finished (`<|eos|>`).
    pub fn update(&mut self, tokenizer: &Tokenizer, token: u32) -> Result<bool, Error> {
        if token == EOS {
            return Ok(true);
        }
        let kind = tokenizer.token_type(token)?;
        if kind == TokenType::SubbeatShift {
            if !self.in_shift && self.payload_count > 0 {
                self.generated_events += 1;
                self.payload_count = 0;
                self.last_field = -1;
                self.incomplete = None;
            }
            self.in_shift = true;
            self.shift_run += 1;
            return Ok(false);
        }
        self.in_shift = false;
        self.shift_run = 0;
        self.payload_count += 1;
        let (last, incomplete) = match kind {
            TokenType::Time => (0, None),
            TokenType::Meter => (1, Some(Incomplete::RhythmAfterMeter)),
            TokenType::EighthPosition => (1, None),
            TokenType::Structure => (2, None),
            TokenType::Key => (3, None),
            TokenType::ChordFull => (4, None),
            TokenType::Pitch => (5, Some(Incomplete::MelodyAfterPitch)),
            TokenType::Duration => (5, None),
            other => {
                return Err(Error::Decode(format!(
                    "unexpected prompt token type {:?}",
                    other.name()
                )))
            }
        };
        self.last_field = last;
        self.incomplete = incomplete;
        Ok(false)
    }
}

/// Mask `logits` in place: every id outside `allowed` becomes `-inf`.
pub fn mask_logits(logits: &mut [f32], allowed: &[Range]) {
    let mut cursor = 0usize;
    for run in allowed {
        let start = (run.start as usize).min(logits.len());
        for v in &mut logits[cursor..start] {
            *v = f32::NEG_INFINITY;
        }
        cursor = (run.end as usize).min(logits.len());
    }
    for v in &mut logits[cursor..] {
        *v = f32::NEG_INFINITY;
    }
}

/// `torch.argmax` over masked logits: the **first** maximal index.
pub fn argmax_first(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_value = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_value || (i == 0 && v == best_value) {
            best = i;
            best_value = v;
        }
    }
    best as u32
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    /// Upstream's masks along the committed `synth_full` and `real_full` token oracles
    /// (`testdata/grammar_masks.json`, produced by `native_parity.py grammar` from upstream's own
    /// `PromptGrammarState`). Every step's allowed set must be identical.
    ///
    /// Mutation that must fail: drop the `shift_run < 4` bound, or allow chord tokens after a key
    /// (`last < 4` → `last <= 4`).
    #[test]
    fn masks_match_upstream_on_the_committed_token_oracles() {
        let fixture: Value =
            serde_json::from_str(include_str!("../testdata/grammar_masks.json")).unwrap();
        let tokenizer = Tokenizer::new(300.0, 100, Some("5ba3325af0344c7f")).unwrap();
        let mut checked = 0;
        for (case, data) in fixture["cases"].as_object().unwrap() {
            let mut state = GrammarState::new();
            for step in data["steps"].as_array().unwrap() {
                let expected: Vec<Range> = step["allowed"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|r| Range {
                        start: r[0].as_u64().unwrap() as u32,
                        end: r[1].as_u64().unwrap() as u32,
                    })
                    .collect();
                assert_eq!(state.allowed(&tokenizer), expected, "{case} step {checked}");
                let token = step["token"].as_u64().unwrap() as u32;
                assert!(
                    expected.iter().any(|r| (r.start..r.end).contains(&token)),
                    "{case}: the oracle token {token} is outside the upstream mask"
                );
                checked += 1;
                if state.update(&tokenizer, token).unwrap() {
                    break;
                }
            }
        }
        assert_eq!(checked, 332 + 448);
    }

    #[test]
    fn argmax_takes_the_first_maximum_and_masking_is_total() {
        let mut logits = vec![1.0, 5.0, 5.0, 9.0, 2.0];
        mask_logits(
            &mut logits,
            &[Range { start: 1, end: 3 }, Range { start: 4, end: 5 }],
        );
        assert_eq!(logits[0], f32::NEG_INFINITY);
        assert_eq!(logits[3], f32::NEG_INFINITY);
        assert_eq!(argmax_first(&logits), 1);
    }
}
