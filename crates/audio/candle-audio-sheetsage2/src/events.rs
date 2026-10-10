//! Token sequences ↔ timed events, the window plan and overlap stitching (port of
//! `tokenization_sheetsage2.decode_sequence` / `encode_decoded_sequence` and of the event half of
//! `generation_sheetsage2.py` / `pipeline_sheetsage2.py` at `4f89269`).

use crate::pyfmt::{float_repr, PyJson};
use crate::tokenizer::{Field, TokenType, Tokenizer, DURATION_TEMPLATES, EOS, OUT, SOS};
use crate::Error;

/// One melody note of an event.
#[derive(Clone, Debug, PartialEq)]
pub struct MelodyNote {
    /// MIDI pitch.
    pub pitch: i64,
    /// `0` = vocal, `1` = instrumental.
    pub track: i64,
    /// Duration bin.
    pub duration_bin: u32,
    /// Duration in sub-beat steps.
    pub duration_steps: u32,
    /// Absolute end time, set when the event is stitched into the song.
    pub end_time: Option<f64>,
}

/// One rhythm entry, in token order (upstream builds a dict while iterating the tokens).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RhythmEntry {
    /// `meter`.
    Meter(u32, u32),
    /// `eighth_position`.
    EighthPosition(u32),
}

/// The decoded rhythm payload: at most one meter and one eighth position, keyed in first-seen order
/// with the last value (Python dict semantics).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Rhythm {
    entries: Vec<RhythmEntry>,
}

impl Rhythm {
    fn set(&mut self, entry: RhythmEntry) {
        let same = |e: &RhythmEntry| {
            matches!(
                (e, &entry),
                (RhythmEntry::Meter(..), RhythmEntry::Meter(..))
                    | (
                        RhythmEntry::EighthPosition(..),
                        RhythmEntry::EighthPosition(..)
                    )
            )
        };
        match self.entries.iter_mut().find(|e| same(e)) {
            Some(slot) => *slot = entry,
            None => self.entries.push(entry),
        }
    }

    /// The meter, if present.
    pub fn meter(&self) -> Option<(u32, u32)> {
        self.entries.iter().find_map(|e| match e {
            RhythmEntry::Meter(n, d) => Some((*n, *d)),
            _ => None,
        })
    }

    /// The eighth-note position, if present.
    pub fn eighth_position(&self) -> Option<u32> {
        self.entries.iter().find_map(|e| match e {
            RhythmEntry::EighthPosition(p) => Some(*p),
            _ => None,
        })
    }

    /// `json.dumps`-shaped value.
    pub fn to_json(&self) -> PyJson {
        PyJson::Dict(
            self.entries
                .iter()
                .map(|e| match e {
                    RhythmEntry::Meter(n, d) => (
                        "meter".to_string(),
                        PyJson::List(vec![PyJson::Int(i64::from(*n)), PyJson::Int(i64::from(*d))]),
                    ),
                    RhythmEntry::EighthPosition(p) => {
                        ("eighth_position".to_string(), PyJson::Int(i64::from(*p)))
                    }
                })
                .collect(),
        )
    }

    /// Upstream `field_text` of the rhythm dict (`meter:(4, 4),eighth_position:0`).
    fn field_text(&self) -> String {
        self.entries
            .iter()
            .map(|e| match e {
                RhythmEntry::Meter(n, d) => format!("meter:({n}, {d})"),
                RhythmEntry::EighthPosition(p) => format!("eighth_position:{p}"),
            })
            .collect::<Vec<_>>()
            .join(",")
    }
}

/// The decoded value of each present field.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Values {
    /// Seconds (window-local after decoding, absolute after stitching).
    pub timestamp: Option<f64>,
    /// Meter / eighth position.
    pub rhythm: Option<Rhythm>,
    /// Section label.
    pub structure: Option<String>,
    /// Key (normalized spelling).
    pub key: Option<String>,
    /// Chord label (sharp spelling, as tokenized).
    pub chord: Option<String>,
    /// Melody notes.
    pub melody: Option<Vec<MelodyNote>>,
}

/// Where a stitched event came from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stitch {
    /// Absolute event time in the song.
    pub time: f64,
    /// Window the event was accepted from.
    pub window_index: usize,
    /// Start of that window.
    pub window_start: f64,
    /// Sub-beat within its window.
    pub source_subbeat: i64,
    /// Sub-beat on the song-wide grid.
    pub global_subbeat: i64,
}

/// One decoded event.
#[derive(Clone, Debug, PartialEq)]
pub struct Event {
    /// Cumulative sub-beat position within the sequence.
    pub subbeat: i64,
    /// Tokens per field, indexed by [`Field::index`] (empty = absent).
    pub tokens: [Vec<u32>; 6],
    /// Decoded values of the present fields.
    pub values: Values,
    /// Song placement, once stitched.
    pub stitch: Option<Stitch>,
}

impl Event {
    /// Whether `field` has tokens.
    pub fn has(&self, field: Field) -> bool {
        !self.tokens[field.index()].is_empty()
    }

    /// The stitched absolute time.
    pub fn time(&self) -> f64 {
        self.stitch.expect("event is stitched").time
    }

    fn refresh_values(&mut self, tokenizer: &Tokenizer) -> Result<(), Error> {
        let mut values = Values::default();
        for field in Field::ALL {
            let tokens = &self.tokens[field.index()];
            if tokens.is_empty() {
                continue;
            }
            decode_field(tokenizer, field, tokens, &mut values)?;
        }
        self.values = values;
        Ok(())
    }

    /// `json.dumps`-shaped value (the `events.json` entry).
    pub fn to_json(&self) -> PyJson {
        let mut tokens_by_field = Vec::new();
        for field in Field::ALL {
            let tokens = &self.tokens[field.index()];
            if !tokens.is_empty() {
                tokens_by_field.push((
                    field.name().to_string(),
                    PyJson::List(tokens.iter().map(|&t| PyJson::Int(i64::from(t))).collect()),
                ));
            }
        }
        let mut values = Vec::new();
        for field in Field::ALL {
            if !self.has(field) {
                continue;
            }
            let value = match field {
                Field::Timestamp => PyJson::Float(self.values.timestamp.expect("present")),
                Field::Rhythm => self.values.rhythm.as_ref().expect("present").to_json(),
                Field::Structure => PyJson::Str(self.values.structure.clone().expect("present")),
                Field::Key => PyJson::Str(self.values.key.clone().expect("present")),
                Field::Chord => PyJson::Str(self.values.chord.clone().expect("present")),
                Field::Melody => PyJson::List(
                    self.values
                        .melody
                        .as_ref()
                        .expect("present")
                        .iter()
                        .map(|n| {
                            let mut note = vec![
                                ("pitch".to_string(), PyJson::Int(n.pitch)),
                                ("track".to_string(), PyJson::Int(n.track)),
                                (
                                    "duration_bin".to_string(),
                                    PyJson::Int(i64::from(n.duration_bin)),
                                ),
                                (
                                    "duration_steps".to_string(),
                                    PyJson::Int(i64::from(n.duration_steps)),
                                ),
                            ];
                            if let Some(end) = n.end_time {
                                note.push(("end_time".to_string(), PyJson::Float(end)));
                            }
                            PyJson::Dict(note)
                        })
                        .collect(),
                ),
            };
            values.push((field.name().to_string(), value));
        }
        let mut entry = vec![
            ("subbeat".to_string(), PyJson::Int(self.subbeat)),
            ("tokens_by_field".to_string(), PyJson::Dict(tokens_by_field)),
            ("values".to_string(), PyJson::Dict(values)),
        ];
        if let Some(s) = self.stitch {
            entry.extend([
                ("time".to_string(), PyJson::Float(s.time)),
                (
                    "window_index".to_string(),
                    PyJson::Int(s.window_index as i64),
                ),
                ("window_start".to_string(), PyJson::Float(s.window_start)),
                ("source_subbeat".to_string(), PyJson::Int(s.source_subbeat)),
                ("global_subbeat".to_string(), PyJson::Int(s.global_subbeat)),
            ]);
        }
        PyJson::Dict(entry)
    }

    /// Upstream `field_text` (the `events.tsv` `fields` column).
    pub fn field_text(&self) -> String {
        let mut parts = Vec::new();
        for field in Field::ALL {
            if !self.has(field) {
                continue;
            }
            match field {
                Field::Timestamp => parts.push(format!(
                    "timestamp={}",
                    float_repr(self.values.timestamp.expect("present"))
                )),
                Field::Rhythm => parts.push(format!(
                    "rhythm={}",
                    self.values.rhythm.as_ref().expect("present").field_text()
                )),
                Field::Structure => parts.push(format!(
                    "structure={}",
                    self.values.structure.as_deref().expect("present")
                )),
                Field::Key => parts.push(format!(
                    "key={}",
                    self.values.key.as_deref().expect("present")
                )),
                Field::Chord => parts.push(format!(
                    "chord={}",
                    self.values.chord.as_deref().expect("present")
                )),
                Field::Melody => {
                    let notes: Vec<String> = self
                        .values
                        .melody
                        .as_ref()
                        .expect("present")
                        .iter()
                        .map(|n| {
                            format!(
                                "pitch={}:track={}:dur_bin={}:dur_steps={}",
                                n.pitch, n.track, n.duration_bin, n.duration_steps
                            )
                        })
                        .collect();
                    parts.push(format!("melody=[{}]", notes.join(",")));
                }
            }
        }
        parts.join("; ")
    }
}

/// A decoded (or stitched) sequence.
#[derive(Clone, Debug, PartialEq)]
pub struct Decoded {
    /// The prompts, in schema order.
    pub prompts: Vec<&'static str>,
    /// The events.
    pub events: Vec<Event>,
    /// Whether the sequence ended with `<|eos|>`.
    pub has_eos: bool,
}

impl Decoded {
    /// `events.json` (`json.dumps(decoded, indent=2)`).
    pub fn to_json(&self) -> PyJson {
        PyJson::Dict(vec![
            ("schema_version".into(), PyJson::Str("v1".into())),
            (
                "prompts".into(),
                PyJson::List(
                    self.prompts
                        .iter()
                        .map(|p| PyJson::Str((*p).to_string()))
                        .collect(),
                ),
            ),
            (
                "events".into(),
                PyJson::List(self.events.iter().map(Event::to_json).collect()),
            ),
            ("has_eos".into(), PyJson::Bool(self.has_eos)),
        ])
    }
}

fn decode_field(
    tokenizer: &Tokenizer,
    field: Field,
    tokens: &[u32],
    values: &mut Values,
) -> Result<(), Error> {
    match field {
        Field::Timestamp => {
            let id = tokenizer
                .time_id(tokens[0])
                .ok_or_else(|| Error::Decode(format!("token {} is not a time token", tokens[0])))?;
            values.timestamp = Some(f64::from(id) / f64::from(tokenizer.time_hz()));
        }
        Field::Rhythm => {
            let mut rhythm = Rhythm::default();
            for &token in tokens {
                match tokenizer.token_type(token)? {
                    TokenType::Meter => {
                        let (n, d) = tokenizer.meter_of(token);
                        rhythm.set(RhythmEntry::Meter(n, d));
                    }
                    TokenType::EighthPosition => rhythm.set(RhythmEntry::EighthPosition(
                        token - tokenizer.eighth_position.start,
                    )),
                    _ => {}
                }
            }
            values.rhythm = Some(rhythm);
        }
        Field::Structure => {
            values.structure = Some(tokenizer.structure_of(tokens[0]).to_string());
        }
        Field::Key => values.key = Some(tokenizer.key_of(tokens[0])),
        Field::Chord => values.chord = Some(tokenizer.chord_of(tokens[0]).to_string()),
        Field::Melody => {
            let mut notes = Vec::new();
            let mut index = 0;
            while index < tokens.len() {
                // Upstream subtracts the pitch-block start from whatever token is here.
                let pitch_id = i64::from(tokens[index]) - i64::from(tokenizer.pitch.start);
                let mut duration_bin = 0u32;
                if index + 1 < tokens.len()
                    && tokenizer.token_type(tokens[index + 1])? == TokenType::Duration
                {
                    duration_bin = tokens[index + 1] - tokenizer.duration.start;
                    index += 2;
                } else {
                    index += 1;
                }
                notes.push(MelodyNote {
                    pitch: pitch_id.rem_euclid(128),
                    track: i64::from(pitch_id >= 128),
                    duration_bin,
                    duration_steps: DURATION_TEMPLATES[duration_bin as usize],
                    end_time: None,
                });
            }
            values.melody = Some(notes);
        }
    }
    Ok(())
}

/// Upstream `decode_sequence`: parse one prompt-conditioned sequence into typed events.
pub fn decode_sequence(
    tokenizer: &Tokenizer,
    tokens: &[u32],
    strict: bool,
) -> Result<Decoded, Error> {
    let mut tokens = tokens.to_vec();
    while tokens.last() == Some(&crate::tokenizer::PAD) {
        tokens.pop();
    }
    if tokens.first() != Some(&SOS) {
        return Err(Error::Decode("sequence must begin with <|sos|>".into()));
    }
    let out_index = tokens[1..]
        .iter()
        .position(|&t| t == OUT)
        .map(|i| i + 1)
        .ok_or_else(|| Error::Decode("sequence is missing <|out|>".into()))?;
    let mut prompts = Vec::new();
    for &token in &tokens[1..out_index] {
        prompts.push(
            tokenizer
                .prompt_name(token)
                .ok_or_else(|| Error::Decode(format!("token {token} is not a prompt token")))?,
        );
    }
    if strict && tokenizer.normalize_prompts(&prompts)? != prompts {
        return Err(Error::Decode(
            "prompt tokens are not in canonical schema order".into(),
        ));
    }
    let active: Vec<Field> = crate::tokenizer::V1_TASKS
        .iter()
        .filter(|t| prompts.contains(&t.0))
        .map(|t| t.2)
        .collect();

    let mut events = Vec::new();
    let mut position = out_index + 1;
    let mut current_step: i64 = 0;
    let mut saw_eos = false;
    while position < tokens.len() {
        let token = tokens[position];
        if token == EOS {
            saw_eos = true;
            position += 1;
            break;
        }
        if tokenizer.token_type(token)? != TokenType::SubbeatShift {
            return Err(Error::Decode(format!(
                "event at token index {position} has no subbeat shift"
            )));
        }
        let mut shift: i64 = 0;
        while position < tokens.len()
            && tokenizer.token_type(tokens[position])? == TokenType::SubbeatShift
        {
            shift += i64::from(tokens[position] - tokenizer.subbeat_shift.start);
            position += 1;
        }
        current_step += shift;

        let mut by_field: [Vec<u32>; 6] = Default::default();
        while position < tokens.len() {
            let token = tokens[position];
            let kind = tokenizer.token_type(token)?;
            if kind == TokenType::SubbeatShift || token == EOS {
                break;
            }
            let field = kind.field().ok_or_else(|| {
                Error::Decode(format!(
                    "token {token} ({}) has no field in schema v1",
                    kind.name()
                ))
            })?;
            if strict && !active.contains(&field) {
                return Err(Error::Decode(format!(
                    "token {token} belongs to inactive output field {:?}",
                    field.name()
                )));
            }
            by_field[field.index()].push(token);
            position += 1;
        }
        if by_field.iter().all(Vec::is_empty) {
            if strict {
                return Err(Error::Decode(format!(
                    "empty event at subbeat {current_step}"
                )));
            }
            continue;
        }
        if strict {
            validate_payload(tokenizer, &by_field)?;
        }
        let mut event = Event {
            subbeat: current_step,
            tokens: by_field,
            values: Values::default(),
            stitch: None,
        };
        event.refresh_values(tokenizer)?;
        events.push(event);
    }
    if strict && !saw_eos {
        return Err(Error::Decode("sequence is missing <|eos|>".into()));
    }
    if strict && position != tokens.len() {
        return Err(Error::Decode("non-padding tokens follow <|eos|>".into()));
    }
    Ok(Decoded {
        prompts,
        events,
        has_eos: saw_eos,
    })
}

fn validate_payload(tokenizer: &Tokenizer, by_field: &[Vec<u32>; 6]) -> Result<(), Error> {
    for field in Field::ALL {
        let values = &by_field[field.index()];
        if values.is_empty() {
            continue;
        }
        let types = values
            .iter()
            .map(|&t| tokenizer.token_type(t))
            .collect::<Result<Vec<_>, _>>()?;
        match field {
            Field::Timestamp if types != [TokenType::Time] => {
                return Err(Error::Decode(
                    "timestamp event must contain exactly one time token".into(),
                ))
            }
            Field::Rhythm
                if types != [TokenType::EighthPosition]
                    && types != [TokenType::Meter, TokenType::EighthPosition] =>
            {
                let names: Vec<&str> = types.iter().map(|t| t.name()).collect();
                return Err(Error::Decode(format!("invalid rhythm payload: {names:?}")));
            }
            Field::Structure | Field::Key | Field::Chord if values.len() != 1 => {
                return Err(Error::Decode(format!(
                    "field {:?} must contain exactly one token",
                    field.name()
                )))
            }
            Field::Melody => {
                let mut index = 0;
                while index < types.len() {
                    if types[index] != TokenType::Pitch {
                        return Err(Error::Decode(
                            "melody payload must contain pitch tokens with optional duration"
                                .into(),
                        ));
                    }
                    index += if types.get(index + 1) == Some(&TokenType::Duration) {
                        2
                    } else {
                        1
                    };
                }
            }
            _ => {}
        }
    }
    Ok(())
}

/// Upstream `encode_decoded_sequence`: the lossless inverse of [`decode_sequence`].
pub fn encode_decoded_sequence(
    tokenizer: &Tokenizer,
    decoded: &Decoded,
) -> Result<Vec<u32>, Error> {
    let mut output = tokenizer.prompt_prefix(&decoded.prompts)?;
    let mut previous: i64 = 0;
    for event in &decoded.events {
        if event.subbeat < previous {
            return Err(Error::Decode(
                "events must be sorted by non-decreasing subbeat".into(),
            ));
        }
        output.extend(tokenizer.subbeat_shift_tokens((event.subbeat - previous) as u64));
        previous = event.subbeat;
        for field in Field::ALL {
            output.extend_from_slice(&event.tokens[field.index()]);
        }
    }
    if decoded.has_eos {
        output.push(EOS);
    }
    Ok(output)
}

/// Upstream `decode_generated_tokens`: strict decode, falling back to a non-strict decode (with a
/// warning) only for the two recoverable grammar-level defects.
pub fn decode_generated_tokens(
    tokenizer: &Tokenizer,
    tokens: &[u32],
) -> Result<(Decoded, Option<String>), Error> {
    match decode_sequence(tokenizer, tokens, true) {
        Ok(decoded) => Ok((decoded, None)),
        Err(Error::Decode(message))
            if message.contains("empty event at subbeat")
                || message.contains("belongs to inactive output field") =>
        {
            let decoded = decode_sequence(tokenizer, tokens, false)?;
            Ok((decoded, Some(message)))
        }
        Err(other) => Err(other),
    }
}

/// `numpy.interp` for one point (inputs strictly inside the table were checked by the caller).
fn np_interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    let n = xp.len();
    if x == xp[n - 1] {
        return fp[n - 1];
    }
    // Largest j with xp[j] <= x.
    let j = xp.partition_point(|&v| v <= x) - 1;
    if j == n - 1 {
        return fp[j];
    }
    if xp[j] == x {
        return fp[j];
    }
    let slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
    let value = slope * (x - xp[j]) + fp[j];
    if value.is_nan() {
        slope * (x - xp[j + 1]) + fp[j + 1]
    } else {
        value
    }
}

/// `numpy.median` of a non-empty slice.
pub(crate) fn np_median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

/// Upstream `event_time_map`: sub-beat → window-local seconds, interpolated between the decoded
/// timestamps and extrapolated at the median step length beyond them.
#[derive(Clone, Debug)]
pub struct TimeMap {
    steps: Vec<f64>,
    times: Vec<f64>,
    step_seconds: f64,
    target_seconds: f64,
}

impl TimeMap {
    /// Build the map for `decoded` within a window of `target_seconds`.
    pub fn new(decoded: &Decoded, target_seconds: f64) -> Self {
        // dict(anchors): the last timestamp of a sub-beat wins; then sorted by sub-beat.
        let mut anchors: Vec<(i64, f64)> = Vec::new();
        for event in &decoded.events {
            if let Some(t) = event.values.timestamp {
                match anchors.iter_mut().find(|(s, _)| *s == event.subbeat) {
                    Some(slot) => slot.1 = t,
                    None => anchors.push((event.subbeat, t)),
                }
            }
        }
        anchors.sort_by_key(|a| a.0);
        let steps: Vec<f64> = anchors.iter().map(|a| a.0 as f64).collect();
        let times: Vec<f64> = anchors.iter().map(|a| a.1).collect();
        let mut step_seconds = 0.125;
        if anchors.len() >= 2 {
            let ratios: Vec<f64> = steps
                .windows(2)
                .zip(times.windows(2))
                .map(|(s, t)| (t[1] - t[0]) / (s[1] - s[0]).max(1.0))
                .collect();
            let median = np_median(&ratios);
            if median.is_finite() && median > 0.0 {
                step_seconds = median;
            }
        }
        Self {
            steps,
            times,
            step_seconds,
            target_seconds,
        }
    }

    /// Seconds of sub-beat `step`.
    pub fn lookup(&self, step: f64) -> f64 {
        let clip = |v: f64| v.max(0.0).min(self.target_seconds);
        if self.steps.is_empty() {
            return (step * 0.125).max(0.0).min(self.target_seconds);
        }
        let first = self.steps[0];
        let last = *self.steps.last().expect("non-empty");
        if step <= first {
            return clip(self.times[0] + (step - first) * self.step_seconds);
        }
        if step >= last {
            return clip(self.times[self.times.len() - 1] + (step - last) * self.step_seconds);
        }
        np_interp(step, &self.steps, &self.times)
    }
}

/// One window of the whole-song plan (upstream `sliding_window_plan`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Window {
    /// Window start in the song, seconds.
    pub start: f64,
    /// Window end (≤ song duration).
    pub end: f64,
    /// Events at or after this time are accepted from this window.
    pub accept_start: f64,
    /// … and before this time.
    pub accept_end: f64,
    /// End of the overlap region rebuilt as a token prefix.
    pub prefix_end: f64,
    /// Window-local stop time for generation, `None` for the last window.
    pub generation_stop: Option<f64>,
}

/// Upstream `sliding_window_plan`.
pub fn sliding_window_plan(
    duration: f64,
    window_seconds: f64,
    overlap_seconds: f64,
    lookahead_seconds: f64,
) -> Result<Vec<Window>, Error> {
    if !(duration > 0.0 && window_seconds > 0.0) {
        return Err(Error::Request(
            "duration and window length must be positive".into(),
        ));
    }
    if !(0.0 <= lookahead_seconds
        && lookahead_seconds <= overlap_seconds
        && overlap_seconds < window_seconds)
    {
        return Err(Error::Request(
            "require 0 <= lookahead <= overlap < window length".into(),
        ));
    }
    let hop = window_seconds - overlap_seconds;
    let (mut start, mut accepted) = (0.0f64, 0.0f64);
    let mut result = Vec::new();
    loop {
        let last = start + window_seconds >= duration - 1e-6;
        let accept_end = if last {
            duration
        } else {
            start + window_seconds - lookahead_seconds
        };
        result.push(Window {
            start,
            end: duration.min(start + window_seconds),
            accept_start: accepted,
            accept_end,
            prefix_end: accepted,
            generation_stop: (!last).then_some(window_seconds - lookahead_seconds),
        });
        if last {
            return Ok(result);
        }
        accepted = accept_end;
        start = (start + hop).min(duration - window_seconds);
    }
}

/// Upstream `stitched_window_events`: the events of one window accepted into the song, placed on
/// the absolute timeline and the song-wide sub-beat grid.
#[allow(clippy::too_many_arguments)]
pub fn stitched_window_events(
    decoded: &Decoded,
    time_map: &TimeMap,
    window: &Window,
    song_duration: f64,
    window_index: usize,
    global_subbeat_base: i64,
) -> Vec<Event> {
    let eps = 1e-4;
    let mut accepted = Vec::new();
    for event in &decoded.events {
        let local = time_map.lookup(event.subbeat as f64);
        let abs_time = window.start + local;
        if abs_time < window.accept_start - eps {
            continue;
        }
        if abs_time >= window.accept_end - eps || abs_time >= song_duration - eps {
            continue;
        }
        let mut output = event.clone();
        let time = abs_time.max(0.0).min(song_duration);
        output.stitch = Some(Stitch {
            time,
            window_index,
            window_start: window.start,
            source_subbeat: event.subbeat,
            global_subbeat: global_subbeat_base + event.subbeat,
        });
        if output.values.timestamp.is_some() {
            output.values.timestamp = Some(time);
        }
        if let Some(notes) = output.values.melody.as_mut() {
            for note in notes {
                let local_end =
                    time_map.lookup((event.subbeat + i64::from(note.duration_steps)) as f64);
                let end_time = window.start + local_end;
                note.end_time = Some(song_duration.min((time + 0.04).max(end_time)));
            }
        }
        accepted.push(output);
    }
    accepted
}

fn active_context_before(
    events: &[Event],
    tokenizer: &Tokenizer,
    time_abs: f64,
) -> Result<[Option<Vec<u32>>; 4], Error> {
    // structure, key, chord, meter
    let mut state: [Option<Vec<u32>>; 4] = Default::default();
    for event in events {
        let Some(stitch) = event.stitch else {
            continue;
        };
        if stitch.time > time_abs + 1e-6 {
            continue;
        }
        for (slot, field) in [(0, Field::Structure), (1, Field::Key), (2, Field::Chord)] {
            if event.has(field) {
                state[slot] = Some(event.tokens[field.index()].clone());
            }
        }
        let mut meters = Vec::new();
        for &token in &event.tokens[Field::Rhythm.index()] {
            if tokenizer.token_type(token)? == TokenType::Meter {
                meters.push(token);
            }
        }
        if !meters.is_empty() {
            state[3] = Some(vec![meters[0]]);
        }
    }
    Ok(state)
}

/// The overlap prefix of a later window (upstream `build_overlap_prefix_tokens`): the accepted
/// events in `[window_start, prefix_end)`, re-timed into the window and carrying the active
/// structure / key / chord / meter context. Returns `(tokens, base_subbeat)`, or `None` when the
/// overlap holds no beat.
pub fn build_overlap_prefix_tokens(
    stitched: &[Event],
    tokenizer: &Tokenizer,
    prompts: &[&'static str],
    window_start: f64,
    prefix_end: f64,
) -> Result<Option<(Vec<u32>, i64)>, Error> {
    let eps = 1e-4;
    let mut source: Vec<&Event> = stitched
        .iter()
        .filter(|e| {
            let t = e.stitch.map_or(-1.0, |s| s.time);
            window_start - eps <= t && t < prefix_end - eps
        })
        .collect();
    let key = |e: &Event| {
        let s = e.stitch.expect("stitched");
        (s.global_subbeat, s.time)
    };
    source.sort_by(|a, b| {
        let (ka, kb) = (key(a), key(b));
        ka.0.cmp(&kb.0).then(ka.1.total_cmp(&kb.1))
    });
    let Some(first) = source
        .iter()
        .position(|e| e.has(Field::Timestamp) || e.has(Field::Rhythm))
    else {
        return Ok(None);
    };
    let source = &source[first..];
    let base = source[0].stitch.expect("stitched").global_subbeat;
    let context = active_context_before(stitched, tokenizer, source[0].time())?;
    let mut prefix_events = Vec::with_capacity(source.len());
    for event in source {
        let stitch = event.stitch.expect("stitched");
        let mut clone = Event {
            subbeat: (stitch.global_subbeat - base).max(0),
            tokens: event.tokens.clone(),
            values: Values::default(),
            stitch: None,
        };
        if clone.has(Field::Timestamp) {
            let local = (stitch.time - window_start) * f64::from(tokenizer.time_hz());
            let id =
                (local.round_ties_even() as i64).clamp(0, i64::from(tokenizer.n_time_tokens()) - 1);
            clone.tokens[Field::Timestamp.index()] = vec![tokenizer.time_token(id as u32)?];
        }
        clone.refresh_values(tokenizer)?;
        prefix_events.push(clone);
    }
    // apply_prefix_context to the first event.
    let head = &mut prefix_events[0];
    for (slot, field) in [(0, Field::Structure), (1, Field::Key), (2, Field::Chord)] {
        if !head.has(field) {
            if let Some(tokens) = &context[slot] {
                head.tokens[field.index()] = tokens.clone();
            }
        }
    }
    let rhythm = &head.tokens[Field::Rhythm.index()];
    let mut has_meter = false;
    let mut has_eighth = false;
    for &token in rhythm {
        match tokenizer.token_type(token)? {
            TokenType::Meter => has_meter = true,
            TokenType::EighthPosition => has_eighth = true,
            _ => {}
        }
    }
    if has_eighth && !has_meter {
        if let Some(meter) = &context[3] {
            let mut tokens = meter.clone();
            tokens.extend_from_slice(rhythm);
            head.tokens[Field::Rhythm.index()] = tokens;
        }
    }
    head.refresh_values(tokenizer)?;
    let decoded = Decoded {
        prompts: prompts.to_vec(),
        events: prefix_events,
        has_eos: false,
    };
    Ok(Some((encode_decoded_sequence(tokenizer, &decoded)?, base)))
}

/// Upstream `write_window_tokens`: one header line per window, then `index\ttoken\tdescription`.
pub fn tokens_txt(tokenizer: &Tokenizer, windows: &[WindowRecord]) -> Result<String, Error> {
    let mut out = String::new();
    for record in windows {
        let w = &record.window;
        out.push_str(&format!(
            "# window_index={} start={:.6} end={:.6} prefix_end={:.6} accept=[{:.6},{:.6}) \
             generation_stop={} prefix_tokens={} tokens={}\n",
            record.index,
            w.start,
            w.end,
            w.prefix_end,
            w.accept_start,
            w.accept_end,
            w.generation_stop.map_or("None".to_string(), float_repr),
            record.prefix_tokens,
            record.tokens.len()
        ));
        for (index, &token) in record.tokens.iter().enumerate() {
            out.push_str(&format!(
                "{index}\t{token}\t{}\n",
                tokenizer.describe(token)?
            ));
        }
        out.push('\n');
    }
    Ok(out)
}

/// The generated tokens of one window, with the plan entry that produced them.
#[derive(Clone, Debug, PartialEq)]
pub struct WindowRecord {
    /// Window index.
    pub index: usize,
    /// The plan entry.
    pub window: Window,
    /// Length of the overlap prefix the window was conditioned on (0 = none).
    pub prefix_tokens: usize,
    /// Every token, prefix included, ending in `<|eos|>`.
    pub tokens: Vec<u32>,
}

/// Parse a `tokens.txt` produced by [`tokens_txt`] (or by upstream) back into window records.
pub fn parse_tokens_txt(text: &str) -> Result<Vec<WindowRecord>, Error> {
    let bad = |line: &str| Error::Decode(format!("malformed tokens.txt line {line:?}"));
    let mut records: Vec<WindowRecord> = Vec::new();
    for line in text.lines() {
        if line.is_empty() {
            continue;
        }
        if let Some(header) = line.strip_prefix("# ") {
            let field = |name: &str| -> Result<&str, Error> {
                header
                    .split(' ')
                    .find_map(|part| part.strip_prefix(name).and_then(|v| v.strip_prefix('=')))
                    .ok_or_else(|| bad(line))
            };
            let num = |name: &str| -> Result<f64, Error> {
                field(name)?.parse::<f64>().map_err(|_| bad(line))
            };
            let accept = field("accept")?;
            let (a, b) = accept
                .trim_start_matches('[')
                .trim_end_matches(')')
                .split_once(',')
                .ok_or_else(|| bad(line))?;
            let stop = field("generation_stop")?;
            records.push(WindowRecord {
                index: field("window_index")?.parse().map_err(|_| bad(line))?,
                window: Window {
                    start: num("start")?,
                    end: num("end")?,
                    accept_start: a.parse().map_err(|_| bad(line))?,
                    accept_end: b.parse().map_err(|_| bad(line))?,
                    prefix_end: num("prefix_end")?,
                    generation_stop: if stop == "None" {
                        None
                    } else {
                        Some(stop.parse().map_err(|_| bad(line))?)
                    },
                },
                prefix_tokens: field("prefix_tokens")?.parse().map_err(|_| bad(line))?,
                tokens: Vec::new(),
            });
            continue;
        }
        let record = records.last_mut().ok_or_else(|| bad(line))?;
        let token = line
            .split('\t')
            .nth(1)
            .and_then(|t| t.parse::<u32>().ok())
            .ok_or_else(|| bad(line))?;
        record.tokens.push(token);
    }
    Ok(records)
}
