//! A pure, incremental JSON-validity state machine for grammar-constrained decoding (sc-6585).
//!
//! gen-core is tensor-free; this is the host-side policy half of constrained decoding. A token
//! sampler keeps one [`JsonState`] and, at each decode step, asks it which candidate token pieces
//! keep the output a valid JSON *prefix* ([`JsonState::advance`] returns `Some` iff the piece is
//! acceptable) so the rest can be masked. The state also reports [`JsonState::can_stop`] so the
//! sampler only allows the end-of-text token once the JSON value is complete.
//!
//! Scope: GENERIC well-formed JSON (the sc-6585 "minimum" — balanced containers, quoted keys, valid
//! escapes, numbers, literals). It does NOT enforce a specific object SHAPE; the worker's canonical
//! serializer (`serialize_magic_prompt_caption`) imposes the Ideogram caption schema post-hoc. The
//! guarantee here is that the emitted text parses (`serde_json::from_str` succeeds).
//!
//! [`JsonState`] is `Copy` (the open-container stack is packed into a `u64` bit-stack, one bit per
//! depth: 0 = object, 1 = array) so the per-step mask over a 100k+ token vocab does a cheap register
//! copy per candidate instead of a heap allocation.

/// Maximum JSON nesting depth (the bit-stack width). Far beyond any real caption (object → array →
/// object is depth 3); deeper input is rejected as if malformed.
const MAX_DEPTH: u8 = 64;

/// Incremental JSON-prefix validator. Construct with [`JsonState::start`], feed accepted token text
/// with [`JsonState::advance`], and gate the stop token on [`JsonState::can_stop`].
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
    /// Expecting the start of a value (document start, after `:`, or after `,` in an array).
    Value,
    /// After `[` — expecting the first array element or `]`.
    ArrayFirst,
    /// After `{` — expecting the first object key or `}`.
    ObjFirstKey,
    /// After `,` in an object — expecting a key (a trailing comma is rejected).
    ObjKey,
    /// After an object key — expecting `:`.
    Colon,
    /// Inside a string. `key` routes the close transition (key → expect `:`, value → after-value).
    Str { key: bool, esc: Esc },
    /// Inside a number literal; the part tracks whether the number is already complete.
    Num(NumPart),
    /// Matching a `true` / `false` / `null` keyword.
    Lit { word: &'static [u8], pos: u8 },
    /// A value just completed; the enclosing container decides what may follow.
    AfterValue,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Esc {
    /// Normal string body.
    None,
    /// Saw `\`, expecting an escape char.
    Backslash,
    /// Inside `\u`, having consumed `n` of 4 hex digits.
    Unicode(u8),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NumPart {
    /// After a leading `-`, needing the first integer digit.
    Sign,
    /// A leading `0` (no further integer digit allowed).
    IntZero,
    /// One or more integer digits (`1`-`9` start).
    IntDigits,
    /// Just after the `.`, needing the first fraction digit.
    DotFirst,
    /// One or more fraction digits.
    FracDigits,
    /// Just after `e`/`E`, an optional sign or the first exponent digit.
    ExpSign,
    /// After an exponent sign, needing the first exponent digit.
    ExpFirst,
    /// One or more exponent digits.
    ExpDigits,
}

impl NumPart {
    /// Is a number ending in this part a complete, valid number?
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

    /// True iff the JSON value is complete and the document may end here — i.e. the end-of-text token
    /// is allowed. (False before any value, mid-string, mid-container, or mid-incomplete-number.)
    pub fn can_stop(self) -> bool {
        self.depth == 0
            && match self.mode {
                Mode::AfterValue => true,
                Mode::Num(part) => part.complete(),
                _ => false,
            }
    }

    /// Feed one accepted token's decoded text. Returns the resulting state if every char keeps the
    /// output a valid JSON prefix, or `None` if any char would make it un-parseable. Pure: `self` is
    /// unchanged. The sampler uses `state.advance(piece).is_some()` to mask, then assigns the result
    /// for the chosen token.
    pub fn advance(self, piece: &str) -> Option<Self> {
        let mut s = self;
        for c in piece.chars() {
            s.feed(c).ok()?;
        }
        Some(s)
    }

    fn top(self) -> Option<bool> {
        // Some(true) = innermost is an array; Some(false) = object; None = top level.
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
        // Only ever called with depth > 0 (guarded by `top()`).
        self.depth -= 1;
    }

    /// Begin a value with char `c` (shared by `Value` and `ArrayFirst`). Returns Ok(true) if `c`
    /// started a value, Ok(false) if `c` was not a value-start (caller may allow ws/closers),
    /// Err(()) never (non-starts are Ok(false)).
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

    /// What follows a just-completed value, given the enclosing container.
    fn after_value(&mut self, c: char) -> Result<(), ()> {
        match self.top() {
            None => {
                // Top level: only trailing whitespace (then EOS).
                if is_ws(c) {
                    Ok(())
                } else {
                    Err(())
                }
            }
            Some(false) => {
                // Inside an object.
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
                // Inside an array.
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
                    return if self.begin_value(c)? {
                        Ok(())
                    } else {
                        Err(())
                    };
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
                    return if self.begin_value(c)? {
                        Ok(())
                    } else {
                        Err(())
                    };
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
                    // `c` does not extend the number: the number is complete iff it ended on a
                    // complete part; then re-process `c` as what follows a value.
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
                    // Raw control characters must be escaped in JSON strings; forbid them so the
                    // output stays strictly parseable.
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

/// The number-grammar transition: does `c` extend a number currently in `part`, and to what part?
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

    /// Feed a whole string; return the final state if it stays a valid JSON prefix throughout.
    fn run(s: &str) -> Option<JsonState> {
        JsonState::start().advance(s)
    }

    /// A string that is COMPLETE valid JSON: a valid prefix AND can_stop.
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
            // Cross-check against serde to be sure the grammar matches a real parser.
            assert!(
                serde_json::from_str::<serde_json::Value>(s).is_ok(),
                "serde should also accept: {s:?}"
            );
        }
    }

    #[test]
    fn valid_prefixes_are_accepted_but_not_complete() {
        for s in [
            "{",
            "{\"a\"",
            "{\"a\":",
            "{\"a\": 1,",
            "[1,",
            "\"unterminated",
            "-",   // sign, no digit yet
            "1.",  // dot, no fraction digit yet
            "1e",  // exponent, no digit yet
            "tru", // partial literal
            "",    // nothing emitted
        ] {
            assert!(run(s).is_some(), "should be a valid prefix: {s:?}");
            assert!(!is_complete(s), "should NOT be complete: {s:?}");
        }
    }

    #[test]
    fn rejects_malformed_at_the_offending_char() {
        // Each of these is the exact failure shape sc-6585 targets (trailing junk, dropped quote,
        // bad escape, raw control char, trailing comma, double value).
        for s in [
            "{}x",              // trailing junk after a complete value
            "{\"a\": 1} {",     // a second top-level value
            "{a: 1}",           // unquoted key
            "{\"a\": 01}",      // leading-zero number
            "{\"a\": .5}",      // bare-dot number
            "[1,]",             // trailing comma
            "{\"a\": 1,}",      // trailing comma in object
            "{\"a\"\"b\"}",     // missing colon
            "{\"a\": \"\\x\"}", // invalid escape
            "[1 2]",            // missing comma
            "nul", // we only reject on the wrong char, so check a real wrong char below
        ] {
            // For "nul" specifically the prefix is valid; assert the rest are rejected.
            if s == "nul" {
                assert!(run(s).is_some());
                continue;
            }
            assert!(run(s).is_none(), "should be rejected as malformed: {s:?}");
        }
    }

    #[test]
    fn rejects_raw_control_char_in_string() {
        assert!(run("{\"a\": \"line\nbreak\"}").is_none());
        // …but the escaped form is fine.
        assert!(is_complete("{\"a\": \"line\\nbreak\"}"));
    }

    #[test]
    fn advance_is_piecewise_like_multi_char_tokens() {
        // Simulate token pieces that straddle structure (e.g. `"},` as one BPE token).
        let mut st = JsonState::start();
        for piece in ["{", "\"k\"", ":", " \"v\"", "}"] {
            st = st.advance(piece).expect("each piece keeps it valid");
        }
        assert!(st.can_stop(), "object closed → can stop");
        // A piece that internally goes invalid is rejected as a whole.
        assert!(JsonState::start().advance("{}trailing").is_none());
    }

    #[test]
    fn caption_shaped_object_round_trips() {
        let caption = "{\"high_level_description\": \"A red fox.\", \
             \"compositional_deconstruction\": {\"background\": \"snow\", \
             \"elements\": [{\"type\": \"obj\", \"desc\": \"fox\"}]}}";
        assert!(is_complete(caption));
        assert!(serde_json::from_str::<serde_json::Value>(caption).is_ok());
    }
}
