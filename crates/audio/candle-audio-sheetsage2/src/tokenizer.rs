//! The SheetSage2 prompt/event vocabulary (port of `tokenization_sheetsage2.py`, `schema_sheetsage2.py`,
//! `labels_sheetsage2.py` and `durations_sheetsage2.py` at
//! `m-a-p/SheetSage2@4f89269db831bdc1880124164a00d4f9385cd129`).
//!
//! The vocabulary is **computed**, not loaded: prompts, sub-beat shifts, one time token per
//! `1/time_hz` s of the window, meters, eighth-note positions, structure labels, keys, maj/min and
//! full chords, pitches and durations. [`Tokenizer::fingerprint`] reproduces upstream's
//! `tokenizer_fingerprint` (the first 16 hex digits of the SHA-256 of its sorted, compact JSON
//! payload), and [`Tokenizer::new`] refuses a mismatch against the checkpoint's `config.json`, so a
//! drifted vocabulary can never decode a checkpoint's tokens.

use sha2::{Digest, Sha256};

use crate::chord_spelling::normalize_key_name;
use crate::pyfmt::float_repr;
use crate::Error;

/// Schema `v1` task prompts: `(name, sampling group, output field)`, in schema order.
pub const V1_TASKS: [(&str, &str, Field); 8] = [
    ("timestamp", "timestamp", Field::Timestamp),
    ("downbeat_meter", "rhythm", Field::Rhythm),
    ("structure", "structure", Field::Structure),
    ("key", "key", Field::Key),
    ("chord_majmin", "chord", Field::Chord),
    ("chord_full", "chord", Field::Chord),
    ("melody_vocal", "melody", Field::Melody),
    ("melody_full", "melody", Field::Melody),
];

/// The prompts upstream's `transcribe` uses by default (`FULL_TASK_PROMPTS`).
pub const FULL_TASK_PROMPTS: [&str; 6] = [
    "timestamp",
    "downbeat_meter",
    "structure",
    "key",
    "chord_full",
    "melody_full",
];

/// The event fields, in `event_field_order`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Field {
    /// Absolute time (`<time_…>`).
    Timestamp,
    /// Meter and eighth-note position.
    Rhythm,
    /// Section label.
    Structure,
    /// Key.
    Key,
    /// Chord.
    Chord,
    /// Melody notes.
    Melody,
}

impl Field {
    /// Every field in `event_field_order`.
    pub const ALL: [Field; 6] = [
        Field::Timestamp,
        Field::Rhythm,
        Field::Structure,
        Field::Key,
        Field::Chord,
        Field::Melody,
    ];

    /// Upstream's field name.
    pub fn name(self) -> &'static str {
        match self {
            Field::Timestamp => "timestamp",
            Field::Rhythm => "rhythm",
            Field::Structure => "structure",
            Field::Key => "key",
            Field::Chord => "chord",
            Field::Melody => "melody",
        }
    }

    /// Index in `event_field_order`.
    pub fn index(self) -> usize {
        self as usize
    }
}

/// `labels_sheetsage2.STRUCTURE_LABELS` (the second, effective definition).
pub const STRUCTURE_LABELS: [&str; 23] = [
    "silence",
    "intro",
    "outro",
    "verse",
    "chorus",
    "bridge",
    "pre-chorus",
    "post-chorus",
    "interlude",
    "fade-out",
    "loop",
    "rap",
    "preshot",
    "irregular",
    "instrumental",
    "intro and verse",
    "pre-chorus and chorus",
    "verse and pre-chorus",
    "solo",
    "theme",
    "development",
    "variation",
    "pre-outro",
];

/// `durations_sheetsage2.DURATION_TEMPLATES`, in sub-beat steps.
pub const DURATION_TEMPLATES: [u32; 24] = [
    1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048,
    3072, 4096,
];

/// Sharp pitch-class names (`CHROMATIC_SHARPS`).
pub const CHROMATIC_SHARPS: [&str; 12] = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
];

const FULL_CHORD_QUALITIES: [&str; 15] = [
    "maj", "min", "dim", "aug", "maj7", "min7", "7", "hdim7", "dim7", "minmaj7", "sus2", "sus4",
    "sus4(b7)", "maj6", "min6",
];

fn full_chord_inversions(quality: &str) -> &'static [&'static str] {
    match quality {
        "maj" => &["/2", "/3", "/5"],
        "min" => &["/2", "/b3", "/5"],
        "maj7" => &["/3", "/5", "/7"],
        "min7" => &["/b3", "/5", "/b7"],
        "7" => &["/3", "/5", "/b7"],
        _ => &[],
    }
}

/// `FULL_CHORD_VOCABULARY`: `N`, then every quality × root × (inversions, root position).
pub fn full_chord_vocabulary() -> Vec<String> {
    let mut labels = vec!["N".to_string()];
    for quality in FULL_CHORD_QUALITIES {
        for root in CHROMATIC_SHARPS {
            for inversion in full_chord_inversions(quality).iter().chain([&""]) {
                labels.push(format!("{root}:{quality}{inversion}"));
            }
        }
    }
    labels
}

const METER_DENOMINATORS: [u32; 6] = [1, 2, 4, 8, 16, 32];
const PROMPT_CAPACITY: u32 = 256;
const N_EIGHTH_POSITIONS: u32 = 256;
const MAX_SUBBEAT_SHIFT: u32 = 256;

/// Special tokens.
pub const PAD: u32 = 0;
/// `<|sos|>`.
pub const SOS: u32 = 1;
/// `<|eos|>`.
pub const EOS: u32 = 2;
/// `<|out|>`.
pub const OUT: u32 = 3;
const PROMPT_START: u32 = 4;

/// The type of one token id.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenType {
    /// Padding.
    Pad,
    /// `<|sos|>`.
    Sos,
    /// `<|eos|>`.
    Eos,
    /// `<|out|>`.
    Out,
    /// A task prompt.
    Prompt,
    /// A sub-beat shift.
    SubbeatShift,
    /// A time token.
    Time,
    /// A meter.
    Meter,
    /// An eighth-note position.
    EighthPosition,
    /// A structure label.
    Structure,
    /// A key.
    Key,
    /// A maj/min chord.
    ChordMajmin,
    /// A full-vocabulary chord.
    ChordFull,
    /// A pitch (and track).
    Pitch,
    /// A duration bin.
    Duration,
}

impl TokenType {
    /// Upstream's `token_type` string.
    pub fn name(self) -> &'static str {
        match self {
            TokenType::Pad => "pad",
            TokenType::Sos => "sos",
            TokenType::Eos => "eos",
            TokenType::Out => "out",
            TokenType::Prompt => "prompt",
            TokenType::SubbeatShift => "subbeat_shift",
            TokenType::Time => "time",
            TokenType::Meter => "meter",
            TokenType::EighthPosition => "eighth_position",
            TokenType::Structure => "structure",
            TokenType::Key => "key",
            TokenType::ChordMajmin => "chord_majmin",
            TokenType::ChordFull => "chord_full",
            TokenType::Pitch => "pitch",
            TokenType::Duration => "duration",
        }
    }

    /// The output field this token type belongs to, if any.
    pub fn field(self) -> Option<Field> {
        Some(match self {
            TokenType::Time => Field::Timestamp,
            TokenType::Meter | TokenType::EighthPosition => Field::Rhythm,
            TokenType::Structure => Field::Structure,
            TokenType::Key => Field::Key,
            TokenType::ChordMajmin | TokenType::ChordFull => Field::Chord,
            TokenType::Pitch | TokenType::Duration => Field::Melody,
            _ => return None,
        })
    }
}

/// A half-open token-id range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Range {
    /// First id.
    pub start: u32,
    /// One past the last id.
    pub end: u32,
}

impl Range {
    fn contains(self, token: u32) -> bool {
        (self.start..self.end).contains(&token)
    }
}

/// The computed SheetSage2 vocabulary for one window length.
#[derive(Clone, Debug)]
pub struct Tokenizer {
    audio_length_seconds: f64,
    time_hz: u32,
    n_time_tokens: u32,
    /// Sub-beat shifts `0..=256`.
    pub subbeat_shift: Range,
    /// Time tokens.
    pub time: Range,
    /// Meters.
    pub meter: Range,
    /// Eighth-note positions.
    pub eighth_position: Range,
    /// Structure labels.
    pub structure: Range,
    /// Keys.
    pub key: Range,
    /// Maj/min chords.
    pub majmin_chord: Range,
    /// Full chords.
    pub full_chord: Range,
    /// Pitches.
    pub pitch: Range,
    /// Durations.
    pub duration: Range,
    meter_pairs: Vec<(u32, u32)>,
    majmin_labels: Vec<String>,
    full_chord_labels: Vec<String>,
    fingerprint: String,
}

impl Tokenizer {
    /// The vocabulary for a window of `audio_length_seconds` at `time_hz` time tokens per second.
    /// When `expected_fingerprint` is given, a different computed fingerprint is refused.
    pub fn new(
        audio_length_seconds: f64,
        time_hz: u32,
        expected_fingerprint: Option<&str>,
    ) -> Result<Self, Error> {
        let n_time = (audio_length_seconds * f64::from(time_hz)).round();
        if !(n_time >= 1.0 && n_time < f64::from(u32::MAX / 2)) {
            return Err(Error::Config(format!(
                "audio_length_seconds {audio_length_seconds} × time_hz {time_hz} must produce at \
                 least one time token"
            )));
        }
        let n_time_tokens = n_time as u32;
        let prompt_end = PROMPT_START + PROMPT_CAPACITY;
        let range = |start: u32, len: u32| Range {
            start,
            end: start + len,
        };
        let subbeat_shift = range(prompt_end, MAX_SUBBEAT_SHIFT + 1);
        let time = range(subbeat_shift.end, n_time_tokens);
        let meter_pairs: Vec<(u32, u32)> = (1..=32u32)
            .flat_map(|n| METER_DENOMINATORS.iter().map(move |&d| (n, d)))
            .collect();
        let meter = range(time.end, meter_pairs.len() as u32);
        let eighth_position = range(meter.end, N_EIGHTH_POSITIONS);
        let structure = range(eighth_position.end, STRUCTURE_LABELS.len() as u32);
        let key = range(structure.end, 24);
        let majmin_labels: Vec<String> = std::iter::once("N".to_string())
            .chain(CHROMATIC_SHARPS.iter().map(|r| format!("{r}:maj")))
            .chain(CHROMATIC_SHARPS.iter().map(|r| format!("{r}:min")))
            .collect();
        let majmin_chord = range(key.end, majmin_labels.len() as u32);
        let full_chord_labels = full_chord_vocabulary();
        let full_chord = range(majmin_chord.end, full_chord_labels.len() as u32);
        let pitch = range(full_chord.end, 256);
        let duration = range(pitch.end, DURATION_TEMPLATES.len() as u32);
        let mut tokenizer = Self {
            audio_length_seconds,
            time_hz,
            n_time_tokens,
            subbeat_shift,
            time,
            meter,
            eighth_position,
            structure,
            key,
            majmin_chord,
            full_chord,
            pitch,
            duration,
            meter_pairs,
            majmin_labels,
            full_chord_labels,
            fingerprint: String::new(),
        };
        tokenizer.fingerprint = tokenizer.compute_fingerprint();
        if let Some(expected) = expected_fingerprint {
            if expected != tokenizer.fingerprint {
                return Err(Error::Config(format!(
                    "tokenizer fingerprint mismatch for schema v1: expected {expected}, got {}",
                    tokenizer.fingerprint
                )));
            }
        }
        Ok(tokenizer)
    }

    /// Total vocabulary size.
    pub fn n_tokens(&self) -> u32 {
        self.duration.end
    }

    /// Time tokens per second.
    pub fn time_hz(&self) -> u32 {
        self.time_hz
    }

    /// Number of time tokens.
    pub fn n_time_tokens(&self) -> u32 {
        self.n_time_tokens
    }

    /// The window length the vocabulary was built for.
    pub fn audio_length_seconds(&self) -> f64 {
        self.audio_length_seconds
    }

    /// Upstream's `vocab_fingerprint`.
    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    fn compute_fingerprint(&self) -> String {
        // json.dumps(payload, sort_keys=True, separators=(",", ":")): keys sorted, compact.
        let strings = |items: &mut dyn Iterator<Item = String>| {
            let quoted: Vec<String> = items.map(|s| format!("\"{s}\"")).collect();
            format!("[{}]", quoted.join(","))
        };
        let mut payload = String::from("{");
        payload.push_str("\"appended_token_blocks\":[],");
        let _ = std::fmt::Write::write_fmt(
            &mut payload,
            format_args!(
                "\"audio_length_seconds\":{},",
                float_repr(self.audio_length_seconds)
            ),
        );
        payload.push_str(&format!(
            "\"duration_templates\":[{}],",
            DURATION_TEMPLATES.map(|d| d.to_string()).join(",")
        ));
        payload.push_str(&format!(
            "\"event_field_order\":{},",
            strings(&mut Field::ALL.iter().map(|f| f.name().to_string()))
        ));
        payload.push_str(&format!(
            "\"full_chord_labels\":{},",
            strings(&mut self.full_chord_labels.iter().cloned())
        ));
        payload.push_str(&format!(
            "\"majmin_chord_labels\":{},",
            strings(&mut self.majmin_labels.iter().cloned())
        ));
        let meters: Vec<String> = self
            .meter_pairs
            .iter()
            .map(|(n, d)| format!("[{n},{d}]"))
            .collect();
        payload.push_str(&format!("\"meter_pairs\":[{}],", meters.join(",")));
        payload.push_str(&format!("\"n_tokens\":{},", self.n_tokens()));
        payload.push_str(&format!("\"prompt_capacity\":{PROMPT_CAPACITY},"));
        payload.push_str(&format!(
            "\"prompt_names\":{},",
            strings(&mut V1_TASKS.iter().map(|t| t.0.to_string()))
        ));
        payload.push_str("\"schema_version\":\"v1\",");
        payload.push_str(&format!(
            "\"structure_labels\":{},",
            strings(&mut STRUCTURE_LABELS.iter().map(|s| s.to_string()))
        ));
        payload.push_str(&format!("\"time_hz\":{}", self.time_hz));
        payload.push('}');
        let digest = Sha256::digest(payload.as_bytes());
        digest.iter().take(8).map(|b| format!("{b:02x}")).collect()
    }

    /// Prompt token of task `name`.
    pub fn prompt_token(&self, name: &str) -> Option<u32> {
        V1_TASKS
            .iter()
            .position(|t| t.0 == name)
            .map(|i| PROMPT_START + i as u32)
    }

    /// Task name of prompt token `token`.
    pub fn prompt_name(&self, token: u32) -> Option<&'static str> {
        token
            .checked_sub(PROMPT_START)
            .and_then(|i| V1_TASKS.get(i as usize))
            .map(|t| t.0)
    }

    /// Upstream `normalize_prompts`: strip `<|…|>`, dedupe, sort into schema order, and refuse
    /// unknown prompts, two prompts of one sampling group, or none.
    pub fn normalize_prompts<S: AsRef<str>>(
        &self,
        prompts: &[S],
    ) -> Result<Vec<&'static str>, Error> {
        let mut names: Vec<&'static str> = Vec::new();
        for prompt in prompts {
            let mut name = prompt.as_ref().trim();
            if name.starts_with("<|") && name.ends_with("|>") && name.len() >= 4 {
                name = &name[2..name.len() - 2];
            }
            let task = V1_TASKS
                .iter()
                .find(|t| t.0 == name)
                .ok_or_else(|| Error::Request(format!("unknown prompt {:?}", prompt.as_ref())))?;
            if !names.contains(&task.0) {
                names.push(task.0);
            }
        }
        names.sort_by_key(|n| V1_TASKS.iter().position(|t| t.0 == *n));
        let mut groups: Vec<(&str, &str)> = Vec::new();
        for name in &names {
            let task = V1_TASKS.iter().find(|t| t.0 == *name).expect("validated");
            if let Some((_, previous)) = groups.iter().find(|(g, _)| *g == task.1) {
                return Err(Error::Request(format!(
                    "prompts {previous:?} and {name:?} are mutually exclusive within sampling \
                     group {:?}",
                    task.1
                )));
            }
            groups.push((task.1, name));
        }
        if names.is_empty() {
            return Err(Error::Request(
                "at least one task prompt is required".into(),
            ));
        }
        Ok(names)
    }

    /// `<|sos|> prompts… <|out|>` for normalized `prompts`.
    pub fn prompt_prefix(&self, prompts: &[&str]) -> Result<Vec<u32>, Error> {
        let prompts = self.normalize_prompts(prompts)?;
        let mut prefix = vec![SOS];
        prefix.extend(prompts.iter().map(|p| self.prompt_token(p).expect("known")));
        prefix.push(OUT);
        Ok(prefix)
    }

    /// Upstream `token_type`; an id outside the vocabulary (or an unassigned prompt slot) is an
    /// error.
    pub fn token_type(&self, token: u32) -> Result<TokenType, Error> {
        for (kind, range) in [
            (TokenType::SubbeatShift, self.subbeat_shift),
            (TokenType::Time, self.time),
            (TokenType::Meter, self.meter),
            (TokenType::EighthPosition, self.eighth_position),
            (TokenType::Structure, self.structure),
            (TokenType::Key, self.key),
            (TokenType::ChordMajmin, self.majmin_chord),
            (TokenType::ChordFull, self.full_chord),
            (TokenType::Pitch, self.pitch),
            (TokenType::Duration, self.duration),
        ] {
            if range.contains(token) {
                return Ok(kind);
            }
        }
        if self.prompt_name(token).is_some() {
            return Ok(TokenType::Prompt);
        }
        match token {
            PAD => Ok(TokenType::Pad),
            SOS => Ok(TokenType::Sos),
            EOS => Ok(TokenType::Eos),
            OUT => Ok(TokenType::Out),
            _ => Err(Error::Decode(format!(
                "token {token} is outside vocabulary size {}",
                self.n_tokens()
            ))),
        }
    }

    /// Time token → seconds id.
    pub fn time_id(&self, token: u32) -> Option<u32> {
        self.time.contains(token).then(|| token - self.time.start)
    }

    /// Time id → token (refusing ids outside the window).
    pub fn time_token(&self, time_id: u32) -> Result<u32, Error> {
        if time_id >= self.n_time_tokens {
            return Err(Error::Decode(format!(
                "time id {time_id} is outside [0, {})",
                self.n_time_tokens
            )));
        }
        Ok(self.time.start + time_id)
    }

    /// Sub-beat shift → tokens (runs of the maximum shift, then the remainder).
    pub fn subbeat_shift_tokens(&self, mut shift: u64) -> Vec<u32> {
        let mut tokens = Vec::new();
        while shift > u64::from(MAX_SUBBEAT_SHIFT) {
            tokens.push(self.subbeat_shift.start + MAX_SUBBEAT_SHIFT);
            shift -= u64::from(MAX_SUBBEAT_SHIFT);
        }
        tokens.push(self.subbeat_shift.start + shift as u32);
        tokens
    }

    /// Meter of a meter token.
    pub fn meter_of(&self, token: u32) -> (u32, u32) {
        self.meter_pairs[(token - self.meter.start) as usize]
    }

    /// Structure label of a structure token.
    pub fn structure_of(&self, token: u32) -> &'static str {
        STRUCTURE_LABELS[(token - self.structure.start) as usize]
    }

    /// Normalized key name of a key token (`C:major`, `Db:major`, `C#:minor`, …).
    pub fn key_of(&self, token: u32) -> String {
        let id = token - self.key.start;
        let mode = if id >= 12 { "minor" } else { "major" };
        normalize_key_name(&format!("{}:{mode}", CHROMATIC_SHARPS[(id % 12) as usize]))
            .expect("tokenizer keys are always major/minor")
    }

    /// Chord label of a chord token.
    pub fn chord_of(&self, token: u32) -> &str {
        if self.majmin_chord.contains(token) {
            &self.majmin_labels[(token - self.majmin_chord.start) as usize]
        } else {
            &self.full_chord_labels[(token - self.full_chord.start) as usize]
        }
    }

    /// The full chord vocabulary, index-aligned with the full-chord token block.
    pub fn full_chord_labels(&self) -> &[String] {
        &self.full_chord_labels
    }

    /// Upstream `describe`: the human-readable token name used in `tokens.txt`.
    pub fn describe(&self, token: u32) -> Result<String, Error> {
        let kind = self.token_type(token)?;
        Ok(match kind {
            TokenType::Prompt => format!("<|{}|>", self.prompt_name(token).expect("prompt")),
            TokenType::SubbeatShift => {
                format!("<subbeat_shift_{}>", token - self.subbeat_shift.start)
            }
            TokenType::Time => {
                let id = token - self.time.start;
                format!("<time_{:.2}s>", f64::from(id) / f64::from(self.time_hz))
            }
            TokenType::Meter => {
                let (n, d) = self.meter_of(token);
                format!("<meter_{n}/{d}>")
            }
            TokenType::EighthPosition => {
                format!("<eighth_pos_{}>", token - self.eighth_position.start)
            }
            TokenType::Structure => format!("<structure_{}>", self.structure_of(token)),
            TokenType::Key => format!("<key_{}>", self.key_of(token)),
            TokenType::ChordMajmin => format!("<chord_majmin_{}>", self.chord_of(token)),
            TokenType::ChordFull => format!("<chord_full_{}>", self.chord_of(token)),
            TokenType::Pitch => {
                let p = token - self.pitch.start;
                format!("<pitch_{}_track_{}>", p % 128, u32::from(p >= 128))
            }
            TokenType::Duration => format!("<duration_{}>", token - self.duration.start),
            other => format!("<|{}|>", other.name()),
        })
    }
}

#[cfg(test)]
mod tests;
