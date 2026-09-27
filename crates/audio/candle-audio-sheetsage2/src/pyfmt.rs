//! Python-compatible text formatting for the byte-exact exports.
//!
//! Upstream writes its LAB rows with `str(value)` and its JSON with `json.dumps`, so a float is
//! written as CPython's shortest round-trip `repr` (`0.59`, `39.4`, `1e-05`, `300.0`). The native
//! exports reproduce that byte for byte, which is what lets the committed reference outputs be
//! compared exactly rather than approximately.

use std::fmt::Write as _;

/// CPython `repr(float)`: the shortest digits that round-trip, in fixed notation when the decimal
/// exponent is in `[-4, 16)` and in `d.ddde±XX` notation otherwise, always with a fractional part in
/// fixed notation.
pub fn float_repr(value: f64) -> String {
    if value.is_nan() {
        return "nan".into();
    }
    if value.is_infinite() {
        return if value > 0.0 { "inf" } else { "-inf" }.into();
    }
    if value == 0.0 {
        return if value.is_sign_negative() {
            "-0.0"
        } else {
            "0.0"
        }
        .into();
    }
    // Rust's `{:e}` prints the shortest round-trip digits too; only the layout differs.
    let sci = format!("{:e}", value.abs());
    let (mantissa, exponent) = sci.split_once('e').expect("`{:e}` always has an exponent");
    let exponent: i32 = exponent.parse().expect("`{:e}` exponent is an integer");
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    // Position of the decimal point relative to the digit string (`0.d1d2… × 10^decpt`).
    let decpt = exponent + 1;
    let mut out = String::new();
    if value < 0.0 {
        out.push('-');
    }
    if !(-3..=16).contains(&decpt) {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        let _ = write!(
            out,
            "e{}{:02}",
            if exponent < 0 { '-' } else { '+' },
            exponent.abs()
        );
    } else if decpt <= 0 {
        out.push_str("0.");
        out.extend(std::iter::repeat_n('0', (-decpt) as usize));
        out.push_str(&digits);
    } else if (decpt as usize) < digits.len() {
        out.push_str(&digits[..decpt as usize]);
        out.push('.');
        out.push_str(&digits[decpt as usize..]);
    } else {
        out.push_str(&digits);
        out.extend(std::iter::repeat_n('0', decpt as usize - digits.len()));
        out.push_str(".0");
    }
    out
}

/// A JSON value in Python's data model, serialized exactly as `json.dumps` does (key order is
/// insertion order; floats are [`float_repr`]; non-finite floats are `NaN` / `Infinity`).
#[derive(Clone, Debug, PartialEq)]
pub enum PyJson {
    /// `null`.
    Null,
    /// `true` / `false`.
    Bool(bool),
    /// An integer.
    Int(i64),
    /// A float.
    Float(f64),
    /// A string (ASCII-escaped like `ensure_ascii=True`).
    Str(String),
    /// A list or tuple.
    List(Vec<PyJson>),
    /// A dict, in insertion order.
    Dict(Vec<(String, PyJson)>),
}

impl PyJson {
    /// `json.dumps(value)` with the default separators.
    pub fn dumps(&self) -> String {
        let mut out = String::new();
        self.write(&mut out, None, 0);
        out
    }

    /// `json.dumps(value, indent=indent)`.
    pub fn dumps_indent(&self, indent: usize) -> String {
        let mut out = String::new();
        self.write(&mut out, Some(indent), 0);
        out
    }

    fn write(&self, out: &mut String, indent: Option<usize>, level: usize) {
        match self {
            PyJson::Null => out.push_str("null"),
            PyJson::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            PyJson::Int(i) => {
                let _ = write!(out, "{i}");
            }
            PyJson::Float(f) => {
                if f.is_nan() {
                    out.push_str("NaN");
                } else if f.is_infinite() {
                    out.push_str(if *f > 0.0 { "Infinity" } else { "-Infinity" });
                } else {
                    out.push_str(&float_repr(*f));
                }
            }
            PyJson::Str(s) => write_str(out, s),
            PyJson::List(items) => {
                if items.is_empty() {
                    out.push_str("[]");
                    return;
                }
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    separator(out, indent, level + 1, i == 0);
                    item.write(out, indent, level + 1);
                }
                close(out, indent, level);
                out.push(']');
            }
            PyJson::Dict(items) => {
                if items.is_empty() {
                    out.push_str("{}");
                    return;
                }
                out.push('{');
                for (i, (key, value)) in items.iter().enumerate() {
                    separator(out, indent, level + 1, i == 0);
                    write_str(out, key);
                    out.push_str(": ");
                    value.write(out, indent, level + 1);
                }
                close(out, indent, level);
                out.push('}');
            }
        }
    }
}

fn separator(out: &mut String, indent: Option<usize>, level: usize, first: bool) {
    match indent {
        Some(width) => {
            if !first {
                out.push(',');
            }
            out.push('\n');
            out.extend(std::iter::repeat_n(' ', width * level));
        }
        None => {
            if !first {
                out.push_str(", ");
            }
        }
    }
}

fn close(out: &mut String, indent: Option<usize>, level: usize) {
    if let Some(width) = indent {
        out.push('\n');
        out.extend(std::iter::repeat_n(' ', width * level));
    }
}

fn write_str(out: &mut String, s: &str) {
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
            c if (c as u32) < 0x20 || (c as u32) > 0x7e => {
                let mut buf = [0u16; 2];
                for unit in c.encode_utf16(&mut buf) {
                    let _ = write!(out, "\\u{unit:04x}");
                }
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Python `str()` of a value as it appears in a LAB row.
#[derive(Clone, Debug, PartialEq)]
pub enum Cell {
    /// A float (`repr`).
    Float(f64),
    /// An integer.
    Int(i64),
    /// Text, verbatim.
    Text(String),
}

impl Cell {
    /// The cell's text.
    pub fn render(&self) -> String {
        match self {
            Cell::Float(f) => float_repr(*f),
            Cell::Int(i) => i.to_string(),
            Cell::Text(t) => t.clone(),
        }
    }
}

/// Upstream `rows_text`: tab-joined `str()` cells, one row per line.
pub fn rows_text(rows: &[Vec<Cell>]) -> String {
    let mut out = String::new();
    for row in rows {
        let cells: Vec<String> = row.iter().map(Cell::render).collect();
        out.push_str(&cells.join("\t"));
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Values whose CPython `repr` is known exactly (checked in CPython 3.11).
    #[test]
    fn float_repr_matches_cpython() {
        for (value, expected) in [
            (0.59, "0.59"),
            (39.4, "39.4"),
            (300.0, "300.0"),
            (1e-5, "1e-05"),
            (1.5e-5, "1.5e-05"),
            (0.0001, "0.0001"),
            (1e15, "1000000000000000.0"),
            (1e16, "1e+16"),
            (1.2345e17, "1.2345e+17"),
            (-2.5, "-2.5"),
            (0.1 + 0.2, "0.30000000000000004"),
            (123456.789, "123456.789"),
            (-0.0, "-0.0"),
            (2.39, "2.39"),
        ] {
            assert_eq!(float_repr(value), expected, "{value:?}");
        }
    }

    #[test]
    fn dumps_matches_python_layouts() {
        let value = PyJson::Dict(vec![
            (
                "meter".into(),
                PyJson::List(vec![PyJson::Int(4), PyJson::Int(4)]),
            ),
            ("eighth_position".into(), PyJson::Int(0)),
        ]);
        assert_eq!(value.dumps(), r#"{"meter": [4, 4], "eighth_position": 0}"#);
        assert_eq!(
            value.dumps_indent(2),
            "{\n  \"meter\": [\n    4,\n    4\n  ],\n  \"eighth_position\": 0\n}"
        );
        assert_eq!(PyJson::List(vec![]).dumps_indent(2), "[]");
        assert_eq!(PyJson::Dict(vec![]).dumps(), "{}");
        assert_eq!(PyJson::Str("é\"".into()).dumps(), "\"\\u00e9\\\"\"");
    }
}
