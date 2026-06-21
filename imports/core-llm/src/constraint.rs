//! Constrained-decoding policy.
//!
//! `core-llm` is tensor-free, so this is the **host-side half** of constrained decoding: the
//! constraint *type* a request carries, plus a pure, incremental JSON-validity state machine. A
//! backend's sampler keeps one [`JsonState`] and, each step, asks which candidate token pieces keep
//! the output a valid JSON *prefix* ([`JsonState::advance`] returns `Some` iff acceptable) so the
//! rest can be masked, and gates the stop token on [`JsonState::can_stop`]. Wiring this to the
//! backend's logit masking is story 7166; the policy lives here.
//!
//! The [`JsonState`] machine is ported verbatim from the proven gen-core implementation (it is pure
//! and tensor-free) and cross-checked against `serde_json`.

use std::collections::HashSet;

/// A requested output constraint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Constraint {
    /// Output must be a single well-formed JSON value. (Object *shape* is not enforced — only that
    /// the emitted text parses.)
    Json,
}

/// Per-vocab decode table for constrained sampling: the literal text of each token id, plus the set
/// of special/added ids (never valid as JSON content). Build once via
/// [`Tokenizer::constraint_decode_table`](crate::Tokenizer::constraint_decode_table) and cache.
#[derive(Clone, Debug)]
pub struct ConstraintDecodeTable {
    /// `pieces[id]` is the literal decoded text of token `id` (empty for special/added ids).
    pub pieces: Vec<String>,
    /// Special / added token ids, never valid as JSON content.
    pub special: HashSet<u32>,
}

// ---------------------------------------------------------------------------------------------
// Incremental JSON-validity state machine (ported from gen-core json_constraint.rs, sc-6585).
// ---------------------------------------------------------------------------------------------

/// Maximum JSON nesting depth (the bit-stack width).
const MAX_DEPTH: u8 = 64;

/// Incremental JSON-prefix validator. Construct with [`JsonState::start`], feed accepted token text
/// with [`JsonState::advance`], and gate the stop token on [`JsonState::can_stop`]. `Copy` (the
/// open-container stack is packed into a `u64` bit-stack, one bit per depth) so per-step masking
/// over a 100k+ token vocab is a register copy, not a heap allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JsonState {
    /// Container kinds, innermost at bit `depth-1`: 0 = object, 1 = array.
    stack: u64,
    /// Number of currently-open containers (0..=`MAX_DEPTH`).
    depth: u8,
    mode: Mode,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Value,
    ArrayFirst,
    ObjFirstKey,
    ObjKey,
    Colon,
    Str { key: bool, esc: Esc },
    Num(NumPart),
    Lit { word: &'static [u8], pos: u8 },
    AfterValue,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Esc {
    None,
    Backslash,
    Unicode(u8),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NumPart {
    Sign,
    IntZero,
    IntDigits,
    DotFirst,
    FracDigits,
    ExpSign,
    ExpFirst,
    ExpDigits,
}

impl NumPart {
    fn complete(self) -> bool {
        matches!(
            self,
            NumPart::IntZero | NumPart::IntDigits | NumPart::FracDigits | NumPart::ExpDigits
        )
    }
}

fn is_ws(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\n' | '\r')
}

impl JsonState {
    /// The initial state: expecting a single top-level JSON value, nothing emitted yet.
    pub fn start() -> Self {
        Self {
            stack: 0,
            depth: 0,
            mode: Mode::Value,
        }
    }

    /// True iff the value is complete and the document may end — i.e. the stop token is allowed.
    pub fn can_stop(self) -> bool {
        self.depth == 0
            && match self.mode {
                Mode::AfterValue => true,
                Mode::Num(part) => part.complete(),
                _ => false,
            }
    }

    /// Feed one accepted token's decoded text. Returns the resulting state if every char keeps the
    /// output a valid JSON prefix, else `None`. Pure: `self` is unchanged.
    pub fn advance(self, piece: &str) -> Option<Self> {
        let mut s = self;
        for c in piece.chars() {
            s.feed(c).ok()?;
        }
        Some(s)
    }

    fn top(self) -> Option<bool> {
        (self.depth > 0).then(|| (self.stack >> (self.depth - 1)) & 1 == 1)
    }

    fn push(&mut self, array: bool) -> Result<(), ()> {
        if self.depth >= MAX_DEPTH {
            return Err(());
        }
        if array {
            self.stack |= 1 << self.depth;
        } else {
            self.stack &= !(1 << self.depth);
        }
        self.depth += 1;
        Ok(())
    }

    fn pop(&mut self) {
        self.depth -= 1;
    }

    fn begin_value(&mut self, c: char) -> Result<bool, ()> {
        match c {
            '{' => {
                self.push(false)?;
                self.mode = Mode::ObjFirstKey;
            }
            '[' => {
                self.push(true)?;
                self.mode = Mode::ArrayFirst;
            }
            '"' => {
                self.mode = Mode::Str {
                    key: false,
                    esc: Esc::None,
                }
            }
            '-' => self.mode = Mode::Num(NumPart::Sign),
            '0' => self.mode = Mode::Num(NumPart::IntZero),
            '1'..='9' => self.mode = Mode::Num(NumPart::IntDigits),
            't' => {
                self.mode = Mode::Lit {
                    word: b"true",
                    pos: 1,
                }
            }
            'f' => {
                self.mode = Mode::Lit {
                    word: b"false",
                    pos: 1,
                }
            }
            'n' => {
                self.mode = Mode::Lit {
                    word: b"null",
                    pos: 1,
                }
            }
            _ => return Ok(false),
        }
        Ok(true)
    }

    fn after_value(&mut self, c: char) -> Result<(), ()> {
        match self.top() {
            None => {
                if is_ws(c) {
                    Ok(())
                } else {
                    Err(())
                }
            }
            Some(false) => {
                if is_ws(c) {
                    Ok(())
                } else if c == ',' {
                    self.mode = Mode::ObjKey;
                    Ok(())
                } else if c == '}' {
                    self.pop();
                    self.mode = Mode::AfterValue;
                    Ok(())
                } else {
                    Err(())
                }
            }
            Some(true) => {
                if is_ws(c) {
                    Ok(())
                } else if c == ',' {
                    self.mode = Mode::Value;
                    Ok(())
                } else if c == ']' {
                    self.pop();
                    self.mode = Mode::AfterValue;
                    Ok(())
                } else {
                    Err(())
                }
            }
        }
    }

    fn feed(&mut self, c: char) -> Result<(), ()> {
        loop {
            match self.mode {
                Mode::Value => {
                    if is_ws(c) {
                        return Ok(());
                    }
                    return if self.begin_value(c)? { Ok(()) } else { Err(()) };
                }
                Mode::ArrayFirst => {
                    if is_ws(c) {
                        return Ok(());
                    }
                    if c == ']' {
                        self.pop();
                        self.mode = Mode::AfterValue;
                        return Ok(());
                    }
                    return if self.begin_value(c)? { Ok(()) } else { Err(()) };
                }
                Mode::ObjFirstKey => {
                    if is_ws(c) {
                        return Ok(());
                    }
                    return match c {
                        '"' => {
                            self.mode = Mode::Str {
                                key: true,
                                esc: Esc::None,
                            };
                            Ok(())
                        }
                        '}' => {
                            self.pop();
                            self.mode = Mode::AfterValue;
                            Ok(())
                        }
                        _ => Err(()),
                    };
                }
                Mode::ObjKey => {
                    if is_ws(c) {
                        return Ok(());
                    }
                    return if c == '"' {
                        self.mode = Mode::Str {
                            key: true,
                            esc: Esc::None,
                        };
                        Ok(())
                    } else {
                        Err(())
                    };
                }
                Mode::Colon => {
                    if is_ws(c) {
                        return Ok(());
                    }
                    return if c == ':' {
                        self.mode = Mode::Value;
                        Ok(())
                    } else {
                        Err(())
                    };
                }
                Mode::Str { key, esc } => return self.feed_str(c, key, esc),
                Mode::Num(part) => {
                    if let Some(next) = num_extend(part, c) {
                        self.mode = Mode::Num(next);
                        return Ok(());
                    }
                    if !part.complete() {
                        return Err(());
                    }
                    self.mode = Mode::AfterValue;
                    continue;
                }
                Mode::Lit { word, pos } => {
                    let pos = pos as usize;
                    return if pos < word.len() && (c as u32) == u32::from(word[pos]) {
                        if pos + 1 == word.len() {
                            self.mode = Mode::AfterValue;
                        } else {
                            self.mode = Mode::Lit {
                                word,
                                pos: (pos + 1) as u8,
                            };
                        }
                        Ok(())
                    } else {
                        Err(())
                    };
                }
                Mode::AfterValue => return self.after_value(c),
            }
        }
    }

    fn feed_str(&mut self, c: char, key: bool, esc: Esc) -> Result<(), ()> {
        match esc {
            Esc::None => {
                if c == '"' {
                    self.mode = if key { Mode::Colon } else { Mode::AfterValue };
                    Ok(())
                } else if c == '\\' {
                    self.mode = Mode::Str {
                        key,
                        esc: Esc::Backslash,
                    };
                    Ok(())
                } else if (c as u32) < 0x20 {
                    Err(())
                } else {
                    Ok(())
                }
            }
            Esc::Backslash => {
                if matches!(c, '"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't') {
                    self.mode = Mode::Str {
                        key,
                        esc: Esc::None,
                    };
                    Ok(())
                } else if c == 'u' {
                    self.mode = Mode::Str {
                        key,
                        esc: Esc::Unicode(0),
                    };
                    Ok(())
                } else {
                    Err(())
                }
            }
            Esc::Unicode(n) => {
                if c.is_ascii_hexdigit() {
                    self.mode = if n + 1 == 4 {
                        Mode::Str {
                            key,
                            esc: Esc::None,
                        }
                    } else {
                        Mode::Str {
                            key,
                            esc: Esc::Unicode(n + 1),
                        }
                    };
                    Ok(())
                } else {
                    Err(())
                }
            }
        }
    }
}

fn num_extend(part: NumPart, c: char) -> Option<NumPart> {
    match part {
        NumPart::Sign => match c {
            '0' => Some(NumPart::IntZero),
            '1'..='9' => Some(NumPart::IntDigits),
            _ => None,
        },
        NumPart::IntZero => match c {
            '.' => Some(NumPart::DotFirst),
            'e' | 'E' => Some(NumPart::ExpSign),
            _ => None,
        },
        NumPart::IntDigits => match c {
            '0'..='9' => Some(NumPart::IntDigits),
            '.' => Some(NumPart::DotFirst),
            'e' | 'E' => Some(NumPart::ExpSign),
            _ => None,
        },
        NumPart::DotFirst => match c {
            '0'..='9' => Some(NumPart::FracDigits),
            _ => None,
        },
        NumPart::FracDigits => match c {
            '0'..='9' => Some(NumPart::FracDigits),
            'e' | 'E' => Some(NumPart::ExpSign),
            _ => None,
        },
        NumPart::ExpSign => match c {
            '+' | '-' => Some(NumPart::ExpFirst),
            '0'..='9' => Some(NumPart::ExpDigits),
            _ => None,
        },
        NumPart::ExpFirst => match c {
            '0'..='9' => Some(NumPart::ExpDigits),
            _ => None,
        },
        NumPart::ExpDigits => match c {
            '0'..='9' => Some(NumPart::ExpDigits),
            _ => None,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(s: &str) -> Option<JsonState> {
        JsonState::start().advance(s)
    }

    fn is_complete(s: &str) -> bool {
        run(s).map(JsonState::can_stop).unwrap_or(false)
    }

    #[test]
    fn accepts_complete_documents() {
        for s in [
            "{}",
            "[]",
            "{\"a\":1}",
            "{\"a\": 1, \"b\": [true, false, null]}",
            "[1, 2.5, -3, 1e10, -2.5E-3, 0.0]",
            "\"a string\"",
            "  {\n  \"k\": \"v\"\n}\n",
            "{\"nested\": {\"arr\": [{\"x\": 1}]}}",
            "{\"esc\": \"a\\\"b\\\\c\\n\\u00e9\"}",
            "true",
            "null",
            "-0",
        ] {
            assert!(is_complete(s), "should be complete valid JSON: {s:?}");
            assert!(
                serde_json::from_str::<serde_json::Value>(s).is_ok(),
                "serde should also accept: {s:?}"
            );
        }
    }

    #[test]
    fn valid_prefixes_are_accepted_but_not_complete() {
        for s in [
            "{", "{\"a\"", "{\"a\":", "{\"a\": 1,", "[1,", "\"unterminated", "-", "1.", "1e",
            "tru", "",
        ] {
            assert!(run(s).is_some(), "should be a valid prefix: {s:?}");
            assert!(!is_complete(s), "should NOT be complete: {s:?}");
        }
    }

    #[test]
    fn rejects_malformed() {
        for s in [
            "{}x",
            "{\"a\": 1} {",
            "{a: 1}",
            "{\"a\": 01}",
            "{\"a\": .5}",
            "[1,]",
            "{\"a\": 1,}",
            "{\"a\"\"b\"}",
            "{\"a\": \"\\x\"}",
            "[1 2]",
        ] {
            assert!(run(s).is_none(), "should be rejected: {s:?}");
        }
    }

    #[test]
    fn rejects_raw_control_char_in_string() {
        assert!(run("{\"a\": \"line\nbreak\"}").is_none());
        assert!(is_complete("{\"a\": \"line\\nbreak\"}"));
    }

    #[test]
    fn advance_is_piecewise() {
        let mut st = JsonState::start();
        for piece in ["{", "\"k\"", ":", " \"v\"", "}"] {
            st = st.advance(piece).expect("each piece keeps it valid");
        }
        assert!(st.can_stop());
        assert!(JsonState::start().advance("{}trailing").is_none());
    }
}
