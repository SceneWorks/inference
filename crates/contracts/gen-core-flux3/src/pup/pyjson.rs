//! A minimal JSON value with **Python `json` module semantics** — insertion-ordered objects,
//! `json.loads` acceptance and `json.dumps(value, ensure_ascii=False)` output (the default
//! `', '` / `': '` separators, Python `float.__repr__` numbers, non-ASCII kept raw).
//!
//! The upsampler's box tables are JSON *text* that the frozen `bbox.py` parses, mutates and
//! re-serializes with the standard library, and the result is part of the caption the image
//! model receives, so the bytes matter: `serde_json` prints `{"a":1}` where Python prints
//! `{"a": 1}`, and it does not keep object key order. This module exists so the native port
//! produces the same bytes without pulling a second JSON dependency into the contract crate.
//!
//! Deliberate non-goals, each a divergence only on input the upstream flow never produces:
//! `NaN` / `Infinity` literals (Python accepts them, this parser does not), integers beyond
//! `i64` (Python keeps them exact, this parser reads them as `f64`), and the exponent
//! formatting of floats outside `[1e-4, 1e16)` (boxes live on a 0–1000 grid).

use std::fmt::Write as _;

/// A parsed JSON value. Objects keep their key order (first insertion wins the position, the
/// last value wins the slot — Python `dict` semantics for a duplicate key).
#[derive(Clone, Debug, PartialEq)]
pub enum JsonValue {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Array(Vec<JsonValue>),
    Object(Vec<(String, JsonValue)>),
}

impl JsonValue {
    /// `dict.get(key)` — `None` for a non-object or an absent key. A present JSON `null` is
    /// `Some(&JsonValue::Null)`, which [`get_non_null`](Self::get_non_null) folds with absence.
    pub fn get(&self, key: &str) -> Option<&JsonValue> {
        match self {
            JsonValue::Object(entries) => entries.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// `dict.get(key)` with Python's `None` — an absent key and a JSON `null` are the same thing.
    pub fn get_non_null(&self, key: &str) -> Option<&JsonValue> {
        self.get(key).filter(|v| !matches!(v, JsonValue::Null))
    }

    /// `dict[key] = value`: replaces the value in place when the key exists, else appends.
    pub fn set(&mut self, key: &str, value: JsonValue) {
        if let JsonValue::Object(entries) = self {
            if let Some(slot) = entries.iter_mut().find(|(k, _)| k == key) {
                slot.1 = value;
            } else {
                entries.push((key.to_string(), value));
            }
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            JsonValue::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[JsonValue]> {
        match self {
            JsonValue::Array(items) => Some(items),
            _ => None,
        }
    }

    pub fn is_object(&self) -> bool {
        matches!(self, JsonValue::Object(_))
    }

    /// Python truthiness: `None`, `False`, `0`, `0.0`, `""`, `[]` and `{}` are falsy.
    pub fn is_truthy(&self) -> bool {
        match self {
            JsonValue::Null => false,
            JsonValue::Bool(b) => *b,
            JsonValue::Int(i) => *i != 0,
            JsonValue::Float(f) => *f != 0.0,
            JsonValue::Str(s) => !s.is_empty(),
            JsonValue::Array(a) => !a.is_empty(),
            JsonValue::Object(o) => !o.is_empty(),
        }
    }

    /// The numeric value of an `int` or `float` that is **not** a `bool` (Python's
    /// `isinstance(v, bool) or not isinstance(v, (int, float))` exclusion).
    pub fn as_number(&self) -> Option<f64> {
        match self {
            JsonValue::Int(i) => Some(*i as f64),
            JsonValue::Float(f) => Some(*f),
            _ => None,
        }
    }

    /// Python `str(value)` for the scalar kinds the box rows carry (`str(box.get("id"))`).
    pub fn py_str(&self) -> String {
        match self {
            JsonValue::Null => "None".to_string(),
            JsonValue::Bool(true) => "True".to_string(),
            JsonValue::Bool(false) => "False".to_string(),
            JsonValue::Int(i) => i.to_string(),
            JsonValue::Float(f) => py_float_repr(*f),
            JsonValue::Str(s) => s.clone(),
            other => dumps(other),
        }
    }
}

/// A `json.loads` failure (`json.JSONDecodeError`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JsonError {
    pub message: String,
    /// Byte offset of the failure.
    pub offset: usize,
}

impl std::fmt::Display for JsonError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} at byte {}", self.message, self.offset)
    }
}

impl std::error::Error for JsonError {}

/// `json.loads(text)`: one JSON value, surrounded by at most JSON whitespace.
pub fn loads(text: &str) -> Result<JsonValue, JsonError> {
    let mut parser = Parser {
        bytes: text.as_bytes(),
        text,
        pos: 0,
        depth: 0,
    };
    parser.skip_ws();
    let value = parser.value()?;
    parser.skip_ws();
    if parser.pos != parser.bytes.len() {
        return Err(parser.error("Extra data"));
    }
    Ok(value)
}

/// `json.dumps(value, ensure_ascii=False)` with the default separators.
pub fn dumps(value: &JsonValue) -> String {
    let mut out = String::new();
    write_value(&mut out, value);
    out
}

fn write_value(out: &mut String, value: &JsonValue) {
    match value {
        JsonValue::Null => out.push_str("null"),
        JsonValue::Bool(true) => out.push_str("true"),
        JsonValue::Bool(false) => out.push_str("false"),
        JsonValue::Int(i) => {
            let _ = write!(out, "{i}");
        }
        JsonValue::Float(f) => out.push_str(&py_float_repr(*f)),
        JsonValue::Str(s) => write_string(out, s),
        JsonValue::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_value(out, item);
            }
            out.push(']');
        }
        JsonValue::Object(entries) => {
            out.push('{');
            for (i, (key, item)) in entries.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_string(out, key);
                out.push_str(": ");
                write_value(out, item);
            }
            out.push('}');
        }
    }
}

/// Python's `ensure_ascii=False` string escaping: `"` and `\` are escaped, the C0 controls take
/// their short escapes (`\n`, `\r`, `\t`, `\b`, `\f`) or `\u00XX`; everything else is raw.
fn write_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Python `float.__repr__`: the shortest round-trip digits, positional when the decimal exponent
/// is in `[-4, 16)` (`82.5`, `80.0`, `0.0001`), else scientific with a signed two-digit exponent
/// (`1e+16`, `1e-05`). `nan` / `inf` render as `json.dumps` does (`NaN`, `Infinity`).
pub fn py_float_repr(x: f64) -> String {
    if x.is_nan() {
        return "NaN".to_string();
    }
    if x.is_infinite() {
        return if x > 0.0 { "Infinity" } else { "-Infinity" }.to_string();
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0" } else { "0.0" }.to_string();
    }
    // `{:e}` is the shortest round-trip mantissa: "d[.ddd]e<exp>".
    let sci = format!("{:e}", x.abs());
    let (mantissa, exp) = sci
        .split_once('e')
        .expect("{:e} always carries an exponent");
    let exp: i32 = exp.parse().expect("{:e} exponent is an integer");
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let sign = if x < 0.0 { "-" } else { "" };
    // Python: `decpt` is the position of the decimal point in `digits`; repr switches to
    // scientific when `decpt <= -4 or decpt > 16` (i.e. exp < -4 or exp >= 16).
    if (-4..16).contains(&exp) {
        let decpt = exp + 1;
        let body = if decpt <= 0 {
            format!("0.{}{}", "0".repeat((-decpt) as usize), digits)
        } else if (decpt as usize) >= digits.len() {
            format!("{}{}.0", digits, "0".repeat(decpt as usize - digits.len()))
        } else {
            let (int_part, frac) = digits.split_at(decpt as usize);
            format!("{int_part}.{frac}")
        };
        format!("{sign}{body}")
    } else {
        let (first, rest) = digits.split_at(1);
        let mantissa = if rest.is_empty() {
            first.to_string()
        } else {
            format!("{first}.{rest}")
        };
        let exp_sign = if exp < 0 { "-" } else { "+" };
        format!("{sign}{mantissa}e{exp_sign}{:02}", exp.abs())
    }
}

struct Parser<'a> {
    bytes: &'a [u8],
    text: &'a str,
    pos: usize,
    depth: usize,
}

/// Python's recursion limit would stop a pathological nesting; this is the same guard.
const MAX_DEPTH: usize = 512;

impl Parser<'_> {
    fn error(&self, message: &str) -> JsonError {
        JsonError {
            message: message.to_string(),
            offset: self.pos,
        }
    }

    fn skip_ws(&mut self) {
        while self.pos < self.bytes.len()
            && matches!(self.bytes[self.pos], b' ' | b'\t' | b'\n' | b'\r')
        {
            self.pos += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn value(&mut self) -> Result<JsonValue, JsonError> {
        match self.peek() {
            None => Err(self.error("Expecting value")),
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'"') => self.string().map(JsonValue::Str),
            Some(b't') => self.literal("true", JsonValue::Bool(true)),
            Some(b'f') => self.literal("false", JsonValue::Bool(false)),
            Some(b'n') => self.literal("null", JsonValue::Null),
            Some(b'-' | b'0'..=b'9') => self.number(),
            Some(_) => Err(self.error("Expecting value")),
        }
    }

    fn literal(&mut self, word: &str, value: JsonValue) -> Result<JsonValue, JsonError> {
        if self.bytes[self.pos..].starts_with(word.as_bytes()) {
            self.pos += word.len();
            Ok(value)
        } else {
            Err(self.error("Expecting value"))
        }
    }

    fn enter(&mut self) -> Result<(), JsonError> {
        self.depth += 1;
        if self.depth > MAX_DEPTH {
            return Err(self.error("maximum recursion depth exceeded"));
        }
        Ok(())
    }

    fn object(&mut self) -> Result<JsonValue, JsonError> {
        self.enter()?;
        self.pos += 1; // '{'
        let mut entries: Vec<(String, JsonValue)> = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            self.depth -= 1;
            return Ok(JsonValue::Object(entries));
        }
        loop {
            self.skip_ws();
            if self.peek() != Some(b'"') {
                return Err(self.error("Expecting property name enclosed in double quotes"));
            }
            let key = self.string()?;
            self.skip_ws();
            if self.peek() != Some(b':') {
                return Err(self.error("Expecting ':' delimiter"));
            }
            self.pos += 1;
            self.skip_ws();
            let value = self.value()?;
            if let Some(slot) = entries.iter_mut().find(|(k, _)| *k == key) {
                slot.1 = value;
            } else {
                entries.push((key, value));
            }
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    self.depth -= 1;
                    return Ok(JsonValue::Object(entries));
                }
                _ => return Err(self.error("Expecting ',' delimiter")),
            }
        }
    }

    fn array(&mut self) -> Result<JsonValue, JsonError> {
        self.enter()?;
        self.pos += 1; // '['
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            self.depth -= 1;
            return Ok(JsonValue::Array(items));
        }
        loop {
            self.skip_ws();
            items.push(self.value()?);
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    self.depth -= 1;
                    return Ok(JsonValue::Array(items));
                }
                _ => return Err(self.error("Expecting ',' delimiter")),
            }
        }
    }

    /// A JSON string (`strict=True`: raw control characters are rejected).
    fn string(&mut self) -> Result<String, JsonError> {
        self.pos += 1; // opening quote
        let mut out = String::new();
        loop {
            let rest = &self.text[self.pos..];
            let Some(c) = rest.chars().next() else {
                return Err(self.error("Unterminated string starting at"));
            };
            match c {
                '"' => {
                    self.pos += 1;
                    return Ok(out);
                }
                '\\' => {
                    self.pos += 1;
                    let Some(esc) = self.peek() else {
                        return Err(self.error("Unterminated string starting at"));
                    };
                    self.pos += 1;
                    match esc {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let unit = self.hex4()?;
                            if (0xD800..0xDC00).contains(&unit) {
                                // A high surrogate: pair it when a `\uDC00..` follows, else keep
                                // the lone surrogate as U+FFFD (Python keeps it as a lone
                                // surrogate, which Rust strings cannot hold).
                                if self.bytes[self.pos..].starts_with(b"\\u") {
                                    let save = self.pos;
                                    self.pos += 2;
                                    let low = self.hex4()?;
                                    if (0xDC00..0xE000).contains(&low) {
                                        let cp = 0x10000 + ((unit - 0xD800) << 10) + (low - 0xDC00);
                                        out.push(char::from_u32(cp).unwrap_or('\u{FFFD}'));
                                        continue;
                                    }
                                    self.pos = save;
                                }
                                out.push('\u{FFFD}');
                            } else {
                                out.push(char::from_u32(unit).unwrap_or('\u{FFFD}'));
                            }
                        }
                        _ => return Err(self.error("Invalid \\escape")),
                    }
                }
                c if (c as u32) < 0x20 => return Err(self.error("Invalid control character at")),
                c => {
                    out.push(c);
                    self.pos += c.len_utf8();
                }
            }
        }
    }

    fn hex4(&mut self) -> Result<u32, JsonError> {
        let Some(hex) = self.text.get(self.pos..self.pos + 4) else {
            return Err(self.error("Invalid \\uXXXX escape"));
        };
        let unit =
            u32::from_str_radix(hex, 16).map_err(|_| self.error("Invalid \\uXXXX escape"))?;
        self.pos += 4;
        Ok(unit)
    }

    /// Python's `NUMBER_RE`: `(-?(?:0|[1-9]\d*))(\.\d+)?([eE][-+]?\d+)?`; an `int` without a
    /// fraction or exponent, else a `float`.
    fn number(&mut self) -> Result<JsonValue, JsonError> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        match self.peek() {
            Some(b'0') => self.pos += 1,
            Some(b'1'..=b'9') => {
                while matches!(self.peek(), Some(b'0'..=b'9')) {
                    self.pos += 1;
                }
            }
            _ => {
                self.pos = start;
                return Err(self.error("Expecting value"));
            }
        }
        let mut is_float = false;
        if self.peek() == Some(b'.') && matches!(self.bytes.get(self.pos + 1), Some(b'0'..=b'9')) {
            is_float = true;
            self.pos += 1;
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            let mut probe = self.pos + 1;
            if matches!(self.bytes.get(probe), Some(b'+' | b'-')) {
                probe += 1;
            }
            if matches!(self.bytes.get(probe), Some(b'0'..=b'9')) {
                is_float = true;
                self.pos = probe;
                while matches!(self.peek(), Some(b'0'..=b'9')) {
                    self.pos += 1;
                }
            }
        }
        let lexeme = &self.text[start..self.pos];
        if !is_float {
            if let Ok(i) = lexeme.parse::<i64>() {
                return Ok(JsonValue::Int(i));
            }
        }
        lexeme
            .parse::<f64>()
            .map(JsonValue::Float)
            .map_err(|_| self.error("Invalid number"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dumps_uses_python_separators_and_key_order() {
        let v = loads(r#"{"id": "obj_1", "bbox": [80, 60, 480.5, 380], "desc": "a \"red\" b\n"}"#)
            .unwrap();
        assert_eq!(
            dumps(&v),
            r#"{"id": "obj_1", "bbox": [80, 60, 480.5, 380], "desc": "a \"red\" b\n"}"#
        );
        // Duplicate key: first position, last value (Python dict).
        let v = loads(r#"{"a": 1, "b": 2, "a": 3}"#).unwrap();
        assert_eq!(dumps(&v), r#"{"a": 3, "b": 2}"#);
    }

    #[test]
    fn strings_escape_like_ensure_ascii_false() {
        let v = JsonValue::Str("é\u{1}\u{7f}\t\\/".to_string());
        assert_eq!(dumps(&v), "\"é\\u0001\u{7f}\\t\\\\/\"");
        assert_eq!(
            loads("\"\\u00e9\\ud83d\\ude00\\/\"").unwrap(),
            JsonValue::Str("é😀/".into())
        );
        assert!(
            loads("\"a\nb\"").is_err(),
            "raw control characters are rejected (strict=True)"
        );
    }

    #[test]
    fn floats_render_as_python_repr() {
        for (x, want) in [
            (82.5, "82.5"),
            (80.0, "80.0"),
            (0.5, "0.5"),
            (1000.0, "1000.0"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (-3.25, "-3.25"),
            (0.1 + 0.2, "0.30000000000000004"),
            (123456.789, "123456.789"),
            (1.5e-7, "1.5e-07"),
            (0.0, "0.0"),
        ] {
            assert_eq!(py_float_repr(x), want, "{x}");
        }
    }

    #[test]
    fn numbers_keep_int_vs_float_identity() {
        let v = loads("[80, 80.0, -0, 1e2, 1E-2, 10]").unwrap();
        assert_eq!(
            v,
            JsonValue::Array(vec![
                JsonValue::Int(80),
                JsonValue::Float(80.0),
                JsonValue::Int(0),
                JsonValue::Float(100.0),
                JsonValue::Float(0.01),
                JsonValue::Int(10),
            ])
        );
        assert_eq!(dumps(&v), "[80, 80.0, 0, 100.0, 0.01, 10]");
    }

    #[test]
    fn loads_rejects_trailing_data_and_python_only_literals() {
        assert!(loads("[1] x").is_err());
        assert!(loads("[NaN]").is_err());
        assert!(loads("").is_err());
        assert!(loads("[1,]").is_err());
        assert_eq!(loads(" [ ] ").unwrap(), JsonValue::Array(vec![]));
        assert_eq!(loads("{}").unwrap(), JsonValue::Object(vec![]));
    }

    #[test]
    fn truthiness_and_accessors_follow_python() {
        let v =
            loads(r#"{"a": [], "b": [0], "c": null, "d": "", "e": 0, "f": 0.0, "g": {}}"#).unwrap();
        for key in ["a", "c", "d", "e", "f", "g"] {
            assert!(!v.get(key).unwrap().is_truthy(), "{key}");
        }
        assert!(v.get("b").unwrap().is_truthy());
        assert!(v.get("c").is_some() && v.get_non_null("c").is_none());
        assert!(v.get("zz").is_none());
        assert_eq!(JsonValue::Bool(true).as_number(), None);
        assert_eq!(JsonValue::Int(3).as_number(), Some(3.0));
        assert_eq!(JsonValue::Float(2.5).py_str(), "2.5");
        assert_eq!(JsonValue::Int(5).py_str(), "5");
    }
}
