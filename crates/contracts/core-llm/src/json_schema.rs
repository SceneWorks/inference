//! JSON-Schema-constrained decoding: a compiled schema subset and its incremental state machine.
//!
//! [`JsonSchema::compile`] turns a JSON Schema into a linear program of output segments, and
//! [`SchemaState`] validates emitted text against it one character at a time, the way
//! [`JsonState`](crate::JsonState) validates plain JSON. [`JsonConstraint`](crate::JsonConstraint)
//! drives either machine behind the same masking seam, so a backend that supports
//! [`Constraint::Json`](crate::Constraint::Json) supports [`Constraint::JsonSchema`](crate::Constraint::JsonSchema)
//! with no further code.
//!
//! # Output form
//!
//! The constrained output is **compact** JSON: no whitespace anywhere, and object properties in the
//! schema's `properties` order. That is what XGrammar emits for a strict schema with
//! `any_whitespace` off (SGLang `--constrained-json-disable-any-whitespace`).
//!
//! # Supported subset
//!
//! A schema the subset cannot express is refused at compile time, never approximated:
//!
//! - `type: "object"` with `properties`, a `required` list naming every property, and
//!   `additionalProperties: false`. Optional properties and open objects are refused.
//! - `type: "string"`, optionally with `enum` / `const` (strings only) or `pattern`. A `pattern`
//!   must be anchored (`^…$`) and use literals, `.`, classes (`[a-z]`, `[^…]`, `\d`, `\w`, `\s`),
//!   groups, `|` and the quantifiers `* + ? {n} {n,} {n,m}`, at most [`MAX_PATTERN_STATES`] NFA
//!   states. A pattern-constrained string cannot contain a character JSON must escape.
//! - `type: "boolean"`, `type: "null"`.
//! - `title`, `description` and `$schema` are annotations and are ignored.
//!
//! Any other keyword or type (numbers, arrays, `minLength`, `format`, `anyOf`, `$ref`, …) is
//! refused with an error naming it.

use std::sync::Arc;

use serde_json::{Map, Value};

/// Most NFA states one `pattern` may compile to (the live state set is a `u64` bit set).
pub const MAX_PATTERN_STATES: usize = 64;
/// Most values one `enum` may list (the live candidate set is a `u64` bit set).
pub const MAX_ENUM_VALUES: usize = 64;

/// A compiled JSON Schema (see the module docs for the subset). Cheap to clone; equality is by the
/// source schema.
#[derive(Clone, Debug)]
pub struct JsonSchema {
    source: Value,
    program: Arc<Program>,
}

impl PartialEq for JsonSchema {
    fn eq(&self, other: &Self) -> bool {
        self.source == other.source
    }
}

impl Eq for JsonSchema {}

impl JsonSchema {
    /// Compile `schema`, or say which part of it the subset cannot express.
    pub fn compile(schema: &Value) -> Result<Self, String> {
        let mut segs = Vec::new();
        compile_node(schema, "#", &mut segs)?;
        Ok(Self {
            source: schema.clone(),
            program: Arc::new(Program { segs }),
        })
    }

    /// The schema this was compiled from.
    pub fn source(&self) -> &Value {
        &self.source
    }

    /// The initial state: nothing emitted yet.
    pub fn start(&self) -> SchemaState {
        SchemaState::default().enter(&self.program)
    }

    /// Feed `piece` from `state`: the resulting state if every character keeps the output a valid
    /// prefix of a schema-conforming document, else `None`. Pure.
    pub fn advance(&self, state: SchemaState, piece: &str) -> Option<SchemaState> {
        let mut s = state;
        for c in piece.chars() {
            s = self.program.feed(s, c)?;
        }
        Some(s)
    }

    /// Whether the document is complete in `state` (a stop token is valid here).
    pub fn can_stop(&self, state: SchemaState) -> bool {
        state.seg as usize == self.program.segs.len()
    }

    /// Whether `text` is, in full, a document the schema accepts.
    pub fn accepts(&self, text: &str) -> bool {
        self.advance(self.start(), text)
            .is_some_and(|s| self.can_stop(s))
    }
}

/// The live position inside a [`JsonSchema`]'s program. `Copy`, so masking a large vocabulary
/// copies registers, not heap state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SchemaState {
    /// Index of the current segment (`segs.len()` = complete).
    seg: u32,
    /// Characters consumed inside a literal or enum segment.
    pos: u32,
    /// Live enum candidates or live NFA states.
    set: u64,
    /// String sub-phase: [`OPEN`], [`BODY`], [`ESCAPE`], or `HEX + n` with `n` hex digits left.
    sub: u8,
}

const OPEN: u8 = 0;
const BODY: u8 = 1;
const ESCAPE: u8 = 2;
const HEX: u8 = 3;

#[derive(Debug)]
struct Program {
    segs: Vec<Seg>,
}

#[derive(Debug)]
enum Seg {
    /// Exact text.
    Lit(Vec<char>),
    /// Any JSON string, quotes included.
    Str,
    /// One of these exact JSON texts (each a quoted string or a keyword).
    Enum(Vec<Vec<char>>),
    /// A JSON string whose content matches the NFA.
    Pattern(Nfa),
}

impl Program {
    fn next(&self, s: SchemaState) -> SchemaState {
        SchemaState {
            seg: s.seg + 1,
            ..SchemaState::default()
        }
        .enter(self)
    }

    fn feed(&self, s: SchemaState, c: char) -> Option<SchemaState> {
        let seg = self.segs.get(s.seg as usize)?;
        match seg {
            Seg::Lit(text) => {
                (text.get(s.pos as usize) == Some(&c)).then_some(())?;
                let pos = s.pos + 1;
                Some(if pos as usize == text.len() {
                    self.next(s)
                } else {
                    SchemaState { pos, ..s }
                })
            }
            Seg::Str => match s.sub {
                OPEN => (c == '"').then_some(SchemaState { sub: BODY, ..s }),
                BODY => match c {
                    '"' => Some(self.next(s)),
                    '\\' => Some(SchemaState { sub: ESCAPE, ..s }),
                    c if (c as u32) < 0x20 => None,
                    _ => Some(s),
                },
                ESCAPE => match c {
                    '"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't' => {
                        Some(SchemaState { sub: BODY, ..s })
                    }
                    'u' => Some(SchemaState { sub: HEX + 4, ..s }),
                    _ => None,
                },
                n => {
                    c.is_ascii_hexdigit().then_some(())?;
                    let left = n - HEX - 1;
                    Some(SchemaState {
                        sub: if left == 0 { BODY } else { HEX + left },
                        ..s
                    })
                }
            },
            Seg::Enum(values) => {
                let pos = s.pos as usize;
                let mut alive = 0u64;
                let mut complete = false;
                for (i, v) in values.iter().enumerate() {
                    if s.set & (1 << i) != 0 && v.get(pos) == Some(&c) {
                        alive |= 1 << i;
                        complete |= v.len() == pos + 1;
                    }
                }
                if alive == 0 {
                    return None;
                }
                // Every value is a quoted JSON string (or `true`/`false`), so none is a proper
                // prefix of another: a value that completes here is the only one still alive.
                Some(if complete {
                    self.next(s)
                } else {
                    SchemaState {
                        pos: s.pos + 1,
                        set: alive,
                        ..s
                    }
                })
            }
            Seg::Pattern(nfa) => match s.sub {
                OPEN => (c == '"').then_some(SchemaState {
                    sub: BODY,
                    set: nfa.start,
                    ..s
                }),
                _ => match c {
                    '"' => (s.set & nfa.accept != 0).then(|| self.next(s)),
                    '\\' => None,
                    c if (c as u32) < 0x20 => None,
                    _ => {
                        let set = nfa.step(s.set, c);
                        (set != 0).then_some(SchemaState { set, ..s })
                    }
                },
            },
        }
    }
}

impl SchemaState {
    /// Initialise the per-segment fields on entering a segment.
    fn enter(self, program: &Program) -> Self {
        match program.segs.get(self.seg as usize) {
            Some(Seg::Enum(values)) => Self {
                set: if values.len() == 64 {
                    u64::MAX
                } else {
                    (1u64 << values.len()) - 1
                },
                ..self
            },
            _ => self,
        }
    }
}

fn push_lit(segs: &mut Vec<Seg>, text: &str) {
    if let Some(Seg::Lit(prev)) = segs.last_mut() {
        prev.extend(text.chars());
    } else {
        segs.push(Seg::Lit(text.chars().collect()));
    }
}

const ANNOTATIONS: [&str; 3] = ["title", "description", "$schema"];

fn object_of<'a>(schema: &'a Value, at: &str) -> Result<&'a Map<String, Value>, String> {
    schema
        .as_object()
        .ok_or_else(|| format!("schema {at} is not an object (boolean schemas are not supported)"))
}

fn only_keys(map: &Map<String, Value>, at: &str, allowed: &[&str]) -> Result<(), String> {
    for key in map.keys() {
        if !allowed.contains(&key.as_str()) && !ANNOTATIONS.contains(&key.as_str()) {
            return Err(format!("schema {at}: keyword `{key}` is not supported"));
        }
    }
    Ok(())
}

fn compile_node(schema: &Value, at: &str, segs: &mut Vec<Seg>) -> Result<(), String> {
    let map = object_of(schema, at)?;
    only_keys(
        map,
        at,
        &[
            "type",
            "enum",
            "const",
            "pattern",
            "properties",
            "required",
            "additionalProperties",
        ],
    )?;
    let ty = match map.get("type") {
        Some(Value::String(t)) => Some(t.as_str()),
        None => None,
        Some(other) => {
            return Err(format!(
                "schema {at}: `type` must be one string, got {other}"
            ))
        }
    };
    let choices = match (map.get("enum"), map.get("const")) {
        (Some(_), Some(_)) => {
            return Err(format!("schema {at}: both `enum` and `const`"));
        }
        (Some(Value::Array(values)), None) => Some(values.clone()),
        (Some(_), None) => return Err(format!("schema {at}: `enum` must be an array")),
        (None, Some(value)) => Some(vec![value.clone()]),
        (None, None) => None,
    };
    if let Some(values) = choices {
        if !matches!(ty, None | Some("string")) {
            return Err(format!(
                "schema {at}: `enum`/`const` is supported for strings only"
            ));
        }
        only_keys(map, at, &["type", "enum", "const"])?;
        return compile_enum(&values, at, segs);
    }
    match ty {
        Some("object") => compile_object(map, at, segs),
        Some("string") => {
            only_keys(map, at, &["type", "pattern"])?;
            match map.get("pattern") {
                None => segs.push(Seg::Str),
                Some(Value::String(p)) => segs.push(Seg::Pattern(
                    Nfa::compile(p).map_err(|e| format!("schema {at}: pattern {p:?}: {e}"))?,
                )),
                Some(_) => return Err(format!("schema {at}: `pattern` must be a string")),
            }
            Ok(())
        }
        Some("boolean") => {
            only_keys(map, at, &["type"])?;
            segs.push(Seg::Enum(vec![
                "true".chars().collect(),
                "false".chars().collect(),
            ]));
            Ok(())
        }
        Some("null") => {
            only_keys(map, at, &["type"])?;
            push_lit(segs, "null");
            Ok(())
        }
        Some(other) => Err(format!("schema {at}: type {other:?} is not supported")),
        None => Err(format!("schema {at}: no `type`, `enum` or `const`")),
    }
}

fn compile_enum(values: &[Value], at: &str, segs: &mut Vec<Seg>) -> Result<(), String> {
    if values.is_empty() || values.len() > MAX_ENUM_VALUES {
        return Err(format!(
            "schema {at}: `enum` must list 1 to {MAX_ENUM_VALUES} values, got {}",
            values.len()
        ));
    }
    let mut texts: Vec<Vec<char>> = Vec::new();
    for v in values {
        let Value::String(_) = v else {
            return Err(format!("schema {at}: `enum` value {v} is not a string"));
        };
        let text: Vec<char> = serde_json::to_string(v)
            .expect("a JSON string serializes")
            .chars()
            .collect();
        if !texts.contains(&text) {
            texts.push(text);
        }
    }
    segs.push(Seg::Enum(texts));
    Ok(())
}

fn compile_object(map: &Map<String, Value>, at: &str, segs: &mut Vec<Seg>) -> Result<(), String> {
    only_keys(
        map,
        at,
        &["type", "properties", "required", "additionalProperties"],
    )?;
    if map.get("additionalProperties") != Some(&Value::Bool(false)) {
        return Err(format!(
            "schema {at}: an object must set `additionalProperties: false`"
        ));
    }
    let empty = Map::new();
    let properties = match map.get("properties") {
        None => &empty,
        Some(Value::Object(p)) => p,
        Some(_) => return Err(format!("schema {at}: `properties` must be an object")),
    };
    let required: Vec<&str> = match map.get("required") {
        None => Vec::new(),
        Some(Value::Array(r)) => r
            .iter()
            .map(|v| v.as_str())
            .collect::<Option<_>>()
            .ok_or_else(|| format!("schema {at}: `required` must list strings"))?,
        Some(_) => return Err(format!("schema {at}: `required` must be an array")),
    };
    for name in properties.keys() {
        if !required.contains(&name.as_str()) {
            return Err(format!(
                "schema {at}: property `{name}` is optional; every property must be required"
            ));
        }
    }
    if let Some(extra) = required.iter().find(|r| !properties.contains_key(**r)) {
        return Err(format!(
            "schema {at}: `required` names `{extra}`, which `properties` does not define"
        ));
    }
    push_lit(segs, "{");
    for (i, (name, sub)) in properties.iter().enumerate() {
        let key = serde_json::to_string(name).expect("a JSON string serializes");
        push_lit(segs, &format!("{}{key}:", if i > 0 { "," } else { "" }));
        compile_node(sub, &format!("{at}/properties/{name}"), segs)?;
    }
    push_lit(segs, "}");
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// The `pattern` subset: an anchored regular expression compiled to a Thompson NFA.
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
struct Class {
    ranges: Vec<(char, char)>,
    negated: bool,
}

impl Class {
    fn matches(&self, c: char) -> bool {
        self.ranges.iter().any(|&(lo, hi)| lo <= c && c <= hi) != self.negated
    }
}

#[derive(Clone, Debug)]
enum Ast {
    Class(Class),
    Concat(Vec<Ast>),
    Alt(Vec<Ast>),
    Repeat(Box<Ast>, u32, Option<u32>),
}

struct Parser<'a> {
    chars: std::iter::Peekable<std::str::Chars<'a>>,
}

impl Parser<'_> {
    fn alt(&mut self) -> Result<Ast, String> {
        let mut arms = vec![self.concat()?];
        while self.chars.peek() == Some(&'|') {
            self.chars.next();
            arms.push(self.concat()?);
        }
        Ok(if arms.len() == 1 {
            arms.pop().unwrap()
        } else {
            Ast::Alt(arms)
        })
    }

    fn concat(&mut self) -> Result<Ast, String> {
        let mut items = Vec::new();
        while let Some(&c) = self.chars.peek() {
            if c == '|' || c == ')' {
                break;
            }
            let atom = self.atom()?;
            items.push(self.quantified(atom)?);
        }
        Ok(Ast::Concat(items))
    }

    fn quantified(&mut self, atom: Ast) -> Result<Ast, String> {
        let (min, max) = match self.chars.peek() {
            Some('*') => (0, None),
            Some('+') => (1, None),
            Some('?') => (0, Some(1)),
            Some('{') => {
                self.chars.next();
                let mut body = String::new();
                loop {
                    match self.chars.next() {
                        Some('}') => break,
                        Some(c) => body.push(c),
                        None => return Err("unclosed `{`".into()),
                    }
                }
                let num = |s: &str| {
                    s.parse::<u32>()
                        .map_err(|_| format!("bad repeat {{{body}}}"))
                };
                let (min, max) = match body.split_once(',') {
                    None => (num(&body)?, Some(num(&body)?)),
                    Some((lo, "")) => (num(lo)?, None),
                    Some((lo, hi)) => (num(lo)?, Some(num(hi)?)),
                };
                if max.is_some_and(|m| m < min) {
                    return Err(format!("bad repeat {{{body}}}"));
                }
                return self.after_quantifier(Ast::Repeat(Box::new(atom), min, max));
            }
            _ => return Ok(atom),
        };
        self.chars.next();
        self.after_quantifier(Ast::Repeat(Box::new(atom), min, max))
    }

    fn after_quantifier(&mut self, ast: Ast) -> Result<Ast, String> {
        match self.chars.peek() {
            Some('*' | '+' | '?' | '{') => {
                Err("stacked or lazy quantifiers are not supported".into())
            }
            _ => Ok(ast),
        }
    }

    fn atom(&mut self) -> Result<Ast, String> {
        let c = self.chars.next().ok_or("unexpected end")?;
        Ok(match c {
            '(' => {
                if self.chars.peek() == Some(&'?') {
                    return Err("group modifiers `(?…)` are not supported".into());
                }
                let inner = self.alt()?;
                if self.chars.next() != Some(')') {
                    return Err("unclosed `(`".into());
                }
                inner
            }
            '[' => Ast::Class(self.class()?),
            '.' => Ast::Class(Class {
                ranges: vec![('\n', '\n'), ('\r', '\r')],
                negated: true,
            }),
            '\\' => Ast::Class(self.escape()?),
            '^' | '$' => return Err("anchors are only supported at the ends".into()),
            '*' | '+' | '?' | '{' | '}' | ']' => {
                return Err(format!("unexpected `{c}`"));
            }
            c => Ast::Class(Class {
                ranges: vec![(c, c)],
                negated: false,
            }),
        })
    }

    fn escape(&mut self) -> Result<Class, String> {
        let c = self.chars.next().ok_or("trailing `\\`")?;
        let class = |ranges: Vec<(char, char)>, negated| Class { ranges, negated };
        Ok(match c {
            'd' => class(vec![('0', '9')], false),
            'D' => class(vec![('0', '9')], true),
            'w' => class(vec![('0', '9'), ('A', 'Z'), ('_', '_'), ('a', 'z')], false),
            'W' => class(vec![('0', '9'), ('A', 'Z'), ('_', '_'), ('a', 'z')], true),
            's' => class(vec![('\t', '\r'), (' ', ' ')], false),
            'S' => class(vec![('\t', '\r'), (' ', ' ')], true),
            c if c.is_ascii_alphanumeric() => {
                return Err(format!("escape `\\{c}` is not supported"));
            }
            c => class(vec![(c, c)], false),
        })
    }

    fn class(&mut self) -> Result<Class, String> {
        let negated = self.chars.peek() == Some(&'^');
        if negated {
            self.chars.next();
        }
        let mut ranges = Vec::new();
        let mut first = true;
        loop {
            let c = self.chars.next().ok_or("unclosed `[`")?;
            let lo = match c {
                ']' if !first => break,
                '\\' => {
                    let esc = self.escape()?;
                    if esc.negated {
                        return Err("negated escapes inside `[…]` are not supported".into());
                    }
                    ranges.extend(esc.ranges);
                    first = false;
                    continue;
                }
                c => c,
            };
            first = false;
            let mut lookahead = self.chars.clone();
            if lookahead.next() == Some('-') && lookahead.peek().is_some_and(|&n| n != ']') {
                self.chars.next();
                let hi = match self.chars.next().ok_or("unclosed `[`")? {
                    '\\' => {
                        let esc = self.escape()?;
                        match esc.ranges.as_slice() {
                            [(a, b)] if a == b && !esc.negated => *a,
                            _ => return Err("a class escape cannot end a range".into()),
                        }
                    }
                    c => c,
                };
                if hi < lo {
                    return Err(format!("range {lo}-{hi} is reversed"));
                }
                ranges.push((lo, hi));
            } else {
                ranges.push((lo, lo));
            }
        }
        Ok(Class { ranges, negated })
    }
}

#[derive(Debug)]
struct Nfa {
    /// `class[i]` with `target[i]`: the character transition out of state `i`, if any.
    class: Vec<Option<(Class, usize)>>,
    /// `closure[i]`: the states reachable from `i` by epsilon edges, `i` included.
    closure: Vec<u64>,
    start: u64,
    accept: u64,
}

struct Builder {
    class: Vec<Option<(Class, usize)>>,
    eps: Vec<Vec<usize>>,
}

impl Builder {
    fn state(&mut self) -> Result<usize, String> {
        if self.class.len() == MAX_PATTERN_STATES {
            return Err(format!(
                "compiles to more than {MAX_PATTERN_STATES} NFA states"
            ));
        }
        self.class.push(None);
        self.eps.push(Vec::new());
        Ok(self.class.len() - 1)
    }

    /// Build `ast` from `from`, returning the state it ends in.
    fn build(&mut self, ast: &Ast, from: usize) -> Result<usize, String> {
        match ast {
            Ast::Class(c) => {
                let to = self.state()?;
                self.class[from] = Some((c.clone(), to));
                Ok(to)
            }
            Ast::Concat(items) => {
                let mut at = from;
                for item in items {
                    // A state carries at most one character edge: give each item a fresh entry.
                    let entry = self.state()?;
                    self.eps[at].push(entry);
                    at = self.build(item, entry)?;
                }
                Ok(at)
            }
            Ast::Alt(arms) => {
                let end = self.state()?;
                for arm in arms {
                    let entry = self.state()?;
                    self.eps[from].push(entry);
                    let out = self.build(arm, entry)?;
                    self.eps[out].push(end);
                }
                Ok(end)
            }
            Ast::Repeat(inner, min, max) => {
                let mut at = from;
                for _ in 0..*min {
                    let entry = self.state()?;
                    self.eps[at].push(entry);
                    at = self.build(inner, entry)?;
                }
                match max {
                    None => {
                        let entry = self.state()?;
                        self.eps[at].push(entry);
                        let out = self.build(inner, entry)?;
                        let end = self.state()?;
                        self.eps[at].push(end);
                        self.eps[out].push(entry);
                        self.eps[out].push(end);
                        Ok(end)
                    }
                    Some(max) => {
                        let end = self.state()?;
                        self.eps[at].push(end);
                        for _ in *min..*max {
                            let entry = self.state()?;
                            self.eps[at].push(entry);
                            at = self.build(inner, entry)?;
                            self.eps[at].push(end);
                        }
                        Ok(end)
                    }
                }
            }
        }
    }
}

impl Nfa {
    fn compile(pattern: &str) -> Result<Self, String> {
        let body = pattern
            .strip_prefix('^')
            .and_then(|p| p.strip_suffix('$'))
            .filter(|p| !p.ends_with('\\') || p.ends_with("\\\\"))
            .ok_or("must be anchored as `^…$`")?;
        let mut parser = Parser {
            chars: body.chars().peekable(),
        };
        let ast = parser.alt()?;
        if parser.chars.next().is_some() {
            return Err("unbalanced `)`".into());
        }
        let mut b = Builder {
            class: Vec::new(),
            eps: Vec::new(),
        };
        let start = b.state()?;
        let accept = b.build(&ast, start)?;
        let n = b.class.len();
        let mut closure = vec![0u64; n];
        for (i, slot) in closure.iter_mut().enumerate() {
            let mut stack = vec![i];
            while let Some(s) = stack.pop() {
                if *slot & (1 << s) == 0 {
                    *slot |= 1 << s;
                    stack.extend(&b.eps[s]);
                }
            }
        }
        Ok(Self {
            start: closure[start],
            accept: 1 << accept,
            class: b.class,
            closure,
        })
    }

    fn step(&self, set: u64, c: char) -> u64 {
        let mut out = 0;
        let mut rest = set;
        while rest != 0 {
            let i = rest.trailing_zeros() as usize;
            rest &= rest - 1;
            if let Some((class, to)) = &self.class[i] {
                if class.matches(c) {
                    out |= self.closure[*to];
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn schema(v: Value) -> JsonSchema {
        JsonSchema::compile(&v).unwrap()
    }

    fn t2i() -> JsonSchema {
        schema(json!({
            "type": "object",
            "properties": {
                "expanded_caption": {"type": "string"},
                "aspect_ratio": {"type": "string", "enum": ["1:1", "16:9", "21:9"]}
            },
            "required": ["expanded_caption", "aspect_ratio"],
            "additionalProperties": false
        }))
    }

    #[test]
    fn the_t2i_reply_schema_accepts_only_its_compact_document() {
        let s = t2i();
        assert!(s.accepts(r#"{"expanded_caption":"A cat, \"tabby\" é.","aspect_ratio":"16:9"}"#));
        assert!(s.accepts(r#"{"expanded_caption":"","aspect_ratio":"1:1"}"#));
        for bad in [
            r#"{"expanded_caption": "A","aspect_ratio":"1:1"}"#,
            r#" {"expanded_caption":"A","aspect_ratio":"1:1"}"#,
            r#"{"aspect_ratio":"1:1","expanded_caption":"A"}"#,
            r#"{"expanded_caption":"A","aspect_ratio":"4:3"}"#,
            r#"{"expanded_caption":"A","aspect_ratio":"16:9","x":1}"#,
            r#"{"expanded_caption":"A"}"#,
            "```json\n{}",
            "Here is the caption",
            "{\"expanded_caption\":\"line\nbreak\",\"aspect_ratio\":\"1:1\"}",
            r#"{"expanded_caption":"bad \q","aspect_ratio":"1:1"}"#,
        ] {
            assert!(!s.accepts(bad), "{bad}");
        }
        // A complete document admits nothing more, whitespace included.
        let done = s
            .advance(
                s.start(),
                r#"{"expanded_caption":"A","aspect_ratio":"1:1"}"#,
            )
            .unwrap();
        assert!(s.can_stop(done));
        assert!(s.advance(done, " ").is_none());
    }

    #[test]
    fn enum_values_sharing_a_prefix_stay_open_until_the_closing_quote() {
        let s = t2i();
        let mid = s
            .advance(s.start(), r#"{"expanded_caption":"A","aspect_ratio":"1"#)
            .unwrap();
        assert!(!s.can_stop(mid));
        assert!(s.advance(mid, "6:9\"}").is_some_and(|x| s.can_stop(x)));
        assert!(s.advance(mid, ":1\"}").is_some_and(|x| s.can_stop(x)));
        assert!(s.advance(mid, "0").is_none());
    }

    #[test]
    fn advance_is_piecewise_across_any_split() {
        let s = t2i();
        let doc = r#"{"expanded_caption":"A \"q\" é","aspect_ratio":"21:9"}"#;
        for cut in 0..=doc.len() {
            if !doc.is_char_boundary(cut) {
                continue;
            }
            let half = s.advance(s.start(), &doc[..cut]).unwrap();
            assert!(
                s.advance(half, &doc[cut..]).is_some_and(|x| s.can_stop(x)),
                "{cut}"
            );
        }
    }

    #[test]
    fn patterns_match_anchored_regular_expressions() {
        let ratio = schema(json!({"type": "string", "pattern": "^0*[1-9][0-9]*:0*[1-9][0-9]*$"}));
        for ok in [r#""16:9""#, r#""1:1""#, r#""09:16""#, r#""1024:768""#] {
            assert!(ratio.accepts(ok), "{ok}");
        }
        for bad in [
            r#""0:9""#,
            r#""16 :9""#,
            r#""16:""#,
            r#""adaptive""#,
            r#""16:9:1""#,
            "16:9",
        ] {
            assert!(!ratio.accepts(bad), "{bad}");
        }
        let alt = schema(json!({"type": "string", "pattern": "^(ab|c)+[x-z]?\\d{2,3}$"}));
        for ok in [r#""ab12""#, r#""cabc123""#, r#""cz99""#] {
            assert!(alt.accepts(ok), "{ok}");
        }
        for bad in [r#""ab1""#, r#""12""#, r#""ab1234""#, r#""abx""#] {
            assert!(!alt.accepts(bad), "{bad}");
        }
        let neg = schema(json!({"type": "string", "pattern": "^[^a-c]{1,}$"}));
        assert!(neg.accepts(r#""xyz""#));
        assert!(!neg.accepts(r#""xbz""#));
        // A pattern string never admits a character JSON must escape.
        let any = schema(json!({"type": "string", "pattern": "^.*$"}));
        assert!(!any.accepts(r#""a\"b""#));
    }

    #[test]
    fn const_booleans_null_and_nested_objects_compile() {
        let s = schema(json!({
            "type": "object",
            "properties": {
                "ratio": {"const": "16:9"},
                "ok": {"type": "boolean"},
                "none": {"type": "null"},
                "inner": {"type": "object", "properties": {}, "additionalProperties": false}
            },
            "required": ["ratio", "ok", "none", "inner"],
            "additionalProperties": false
        }));
        assert!(s.accepts(r#"{"ratio":"16:9","ok":false,"none":null,"inner":{}}"#));
        assert!(!s.accepts(r#"{"ratio":"1:1","ok":false,"none":null,"inner":{}}"#));
        assert!(!s.accepts(r#"{"ratio":"16:9","ok":fals,"none":null,"inner":{}}"#));
    }

    #[test]
    fn schemas_outside_the_subset_are_refused_by_name() {
        let refused = |v: Value, needle: &str| {
            let err = JsonSchema::compile(&v).unwrap_err();
            assert!(err.contains(needle), "{err}");
        };
        let obj = |props: Value, required: Value| {
            json!({"type": "object", "properties": props, "required": required,
                   "additionalProperties": false})
        };
        refused(json!({"type": "integer"}), "\"integer\"");
        refused(json!({"type": "string", "minLength": 1}), "minLength");
        refused(
            json!({"type": "object", "properties": {}}),
            "additionalProperties",
        );
        refused(obj(json!({"a": {"type": "string"}}), json!([])), "optional");
        refused(obj(json!({}), json!(["a"])), "does not define");
        refused(json!({"enum": [1, 2]}), "not a string");
        refused(json!({"type": "string", "pattern": "[0-9]+"}), "anchored");
        refused(json!({"type": "string", "pattern": "^a+?$"}), "lazy");
        refused(
            json!({"type": "string", "pattern": "^a{100}$"}),
            "NFA states",
        );
        refused(json!({"anyOf": []}), "anyOf");
    }

    #[test]
    fn equality_is_by_source_schema() {
        assert_eq!(t2i(), t2i());
        assert_ne!(t2i(), schema(json!({"type": "string"})));
    }
}
