//! A Standard MIDI File writer and reader that reproduce `pretty_midi` 0.2.10 + `mido` 1.3.3 exactly
//! for the files the SheetSage2 exports write.
//!
//! Byte parity matters beyond the `.mid` files themselves: upstream's ABC builder reads the notation
//! melody back **through a MIDI round trip** (`notation/song_melody.mid` → `PrettyMIDI`), so every
//! note time the score quantizes is a MIDI-tick-rounded time (960 ticks per quarter at the default
//! 120 BPM = 1/1920 s). Reproducing that rounding is what makes the native ABC byte-identical.

use crate::Error;

/// Seconds per tick at 120 BPM (`60.0 / (120.0 * resolution)`), as `pretty_midi` computes it.
fn tick_scale(resolution: u32, tempo_us: Option<u32>) -> f64 {
    let bpm = tempo_us.map_or(120.0, |t| 6e7 / f64::from(t));
    60.0 / (bpm * f64::from(resolution))
}

/// One note.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Note {
    /// Velocity.
    pub velocity: u8,
    /// MIDI pitch.
    pub pitch: u8,
    /// Start, seconds.
    pub start: f64,
    /// End, seconds.
    pub end: f64,
}

/// One instrument (one MIDI track).
#[derive(Clone, Debug, PartialEq)]
pub struct Instrument {
    /// Program number.
    pub program: u8,
    /// Track name.
    pub name: String,
    /// Notes, in insertion (write) or closing (read) order.
    pub notes: Vec<Note>,
}

/// A `pretty_midi.PrettyMIDI` subset: one tempo, named instruments, notes.
#[derive(Clone, Debug, PartialEq)]
pub struct Midi {
    /// Ticks per quarter note.
    pub resolution: u32,
    /// Microseconds per quarter (`None` = the default 120 BPM of a fresh object).
    pub tempo_us: Option<u32>,
    /// Instruments.
    pub instruments: Vec<Instrument>,
    /// Length of `pretty_midi`'s tick→time table: 1 for a fresh object, `max_tick + 2` after a
    /// load. It decides how [`Midi::to_bytes`] rounds times to ticks.
    tick_table_len: u64,
}

impl Midi {
    /// `PrettyMIDI(resolution=resolution)` (120 BPM).
    pub fn new(resolution: u32) -> Self {
        Self {
            resolution,
            tempo_us: None,
            instruments: Vec::new(),
            tick_table_len: 1,
        }
    }

    fn scale(&self) -> f64 {
        tick_scale(self.resolution, self.tempo_us)
    }

    /// `PrettyMIDI.time_to_tick`: the nearest tick of the tick→time table (`k * scale`), ties to
    /// the later tick; past the table's end, extrapolated and rounded half to even. A fresh object's
    /// table holds only tick 0, so every positive time extrapolates.
    fn time_to_tick(&self, time: f64) -> u64 {
        let scale = self.scale();
        let len = self.tick_table_len;
        let at = |k: u64| scale * k as f64;
        // np.searchsorted(table, time, side="left"): the number of entries < time.
        let mut idx = if time > 0.0 {
            ((time / scale).ceil() as u64).min(len)
        } else {
            0
        };
        while idx > 0 && at(idx - 1) >= time {
            idx -= 1;
        }
        while idx < len && at(idx) < time {
            idx += 1;
        }
        if idx == len {
            let last = len - 1;
            let tick = last as f64 + (time - at(last)) / scale;
            return tick.round_ties_even() as u64;
        }
        if idx > 0 && (time - at(idx - 1)).abs() < (time - at(idx)).abs() {
            idx - 1
        } else {
            idx
        }
    }

    /// `PrettyMIDI.write` → bytes (`mido` SMF type 1, running status, sorted events).
    pub fn to_bytes(&self) -> Vec<u8> {
        let scale = self.scale();
        let tempo = (6e7 / (60.0 / (scale * f64::from(self.resolution)))) as u32;
        let mut tracks = Vec::new();
        // Timing track: set_tempo sorts before the default 4/4 time signature at tick 0.
        let mut timing = Vec::new();
        push_delta(&mut timing, 0);
        timing.extend([0xFF, 0x51, 0x03]);
        timing.extend(&tempo.to_be_bytes()[1..]);
        push_delta(&mut timing, 0);
        timing.extend([0xFF, 0x58, 0x04, 0x04, 0x02, 0x18, 0x08]);
        push_delta(&mut timing, 1);
        timing.extend([0xFF, 0x2F, 0x00]);
        tracks.push(timing);

        let channels: Vec<u8> = (0..16u8).filter(|&c| c != 9).collect();
        for (n, instrument) in self.instruments.iter().enumerate() {
            let channel = channels[n % channels.len()];
            // (tick, sort score, bytes) — pretty_midi's comparator: tick, then type/pitch/velocity.
            let mut events: Vec<(u64, u64, [u8; 3])> = Vec::new();
            events.push((0, 6 * 65536, [0xC0 | channel, instrument.program, 0]));
            for note in &instrument.notes {
                let on = self.time_to_tick(note.start);
                let off = self.time_to_tick(note.end);
                events.push((
                    on,
                    10 * 65536 + u64::from(note.pitch) * 256 + u64::from(note.velocity),
                    [0x90 | channel, note.pitch, note.velocity],
                ));
                events.push((
                    off,
                    10 * 65536 + u64::from(note.pitch) * 256,
                    [0x90 | channel, note.pitch, 0],
                ));
            }
            // A stable sort on (tick, score) is what Python's timsort yields for this comparator.
            events.sort_by_key(|e| (e.0, e.1));
            for i in 0..events.len().saturating_sub(1) {
                let (a, b) = (events[i], events[i + 1]);
                if a.0 == b.0 && a.2[1] == b.2[1] && a.2[2] != 0 && b.2[2] == 0 && a.1 >= 10 * 65536
                {
                    events.swap(i, i + 1);
                }
            }
            let mut data = Vec::new();
            if !instrument.name.is_empty() {
                push_delta(&mut data, 0);
                data.extend([0xFF, 0x03]);
                push_varlen(&mut data, instrument.name.len() as u64);
                data.extend(instrument.name.bytes());
            }
            let mut previous_tick = 0u64;
            let mut running: Option<u8> = None;
            for (tick, _, bytes) in &events {
                push_delta(&mut data, tick - previous_tick);
                previous_tick = *tick;
                let status = bytes[0];
                let body: &[u8] = if status & 0xF0 == 0xC0 {
                    &bytes[..2]
                } else {
                    &bytes[..]
                };
                if running == Some(status) {
                    data.extend(&body[1..]);
                } else {
                    data.extend(body);
                }
                running = Some(status);
            }
            push_delta(&mut data, 1);
            data.extend([0xFF, 0x2F, 0x00]);
            tracks.push(data);
        }
        let mut out = Vec::new();
        out.extend(b"MThd");
        out.extend(6u32.to_be_bytes());
        out.extend(1u16.to_be_bytes());
        out.extend((tracks.len() as u16).to_be_bytes());
        out.extend((self.resolution as u16).to_be_bytes());
        for track in tracks {
            out.extend(b"MTrk");
            out.extend((track.len() as u32).to_be_bytes());
            out.extend(track);
        }
        out
    }

    /// `PrettyMIDI(BytesIO(bytes))` for files written by [`Midi::to_bytes`] (or upstream): tempo,
    /// track names, programs and notes, paired exactly as `pretty_midi._load_instruments` pairs them
    /// (a note-off closes every open note of its pitch that did not start on the same tick; notes
    /// that never close are dropped; instruments appear in the order their first note closes).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, Error> {
        let bad = |what: &str| Error::Symbolic(format!("malformed MIDI: {what}"));
        let mut cursor = Reader { bytes, pos: 0 };
        if cursor.take(4)? != b"MThd" {
            return Err(bad("missing MThd"));
        }
        let header_len = cursor.u32()? as usize;
        let header = cursor.take(header_len)?;
        if header.len() < 6 {
            return Err(bad("short header"));
        }
        let ntracks = u16::from_be_bytes([header[2], header[3]]) as usize;
        let resolution = u32::from(u16::from_be_bytes([header[4], header[5]]));
        let mut raw_tracks = Vec::with_capacity(ntracks);
        for _ in 0..ntracks {
            if cursor.take(4)? != b"MTrk" {
                return Err(bad("missing MTrk"));
            }
            let len = cursor.u32()? as usize;
            raw_tracks.push(parse_track(cursor.take(len)?)?);
        }
        let mut tempo_us = None;
        if let Some(first) = raw_tracks.first() {
            for event in first {
                if let TrackEvent::Tempo(t) = event.kind {
                    if event.tick == 0 {
                        tempo_us = Some(t);
                    } else {
                        return Err(bad("tempo changes after tick 0 are not supported"));
                    }
                }
            }
        }
        let scale = tick_scale(resolution, tempo_us);
        let time = |tick: u64| scale * tick as f64;
        // (program, channel, track) → instrument index, in creation order.
        let mut keys: Vec<(u8, u8, usize)> = Vec::new();
        let mut instruments: Vec<Instrument> = Vec::new();
        for (track_index, events) in raw_tracks.iter().enumerate() {
            let mut name = String::new();
            let mut program = [0u8; 16];
            let mut open: OpenNotes = Vec::new();
            for event in events {
                match event.kind {
                    TrackEvent::Name(ref n) => name = n.clone(),
                    TrackEvent::Program(channel, p) => program[channel as usize] = p,
                    TrackEvent::NoteOn(channel, pitch, velocity) if velocity > 0 => {
                        match open.iter_mut().find(|(k, _)| *k == (channel, pitch)) {
                            Some((_, list)) => list.push((event.tick, velocity)),
                            None => open.push(((channel, pitch), vec![(event.tick, velocity)])),
                        }
                    }
                    TrackEvent::NoteOn(channel, pitch, _) | TrackEvent::NoteOff(channel, pitch) => {
                        let Some(slot) = open.iter().position(|(k, _)| *k == (channel, pitch))
                        else {
                            continue;
                        };
                        let end = event.tick;
                        let (close, keep): (Vec<_>, Vec<_>) =
                            open[slot].1.iter().partition(|(start, _)| *start != end);
                        for (start, velocity) in &close {
                            let key = (program[channel as usize], channel, track_index);
                            let index = match keys.iter().position(|k| *k == key) {
                                Some(i) => i,
                                None => {
                                    keys.push(key);
                                    instruments.push(Instrument {
                                        program: key.0,
                                        name: name.clone(),
                                        notes: Vec::new(),
                                    });
                                    instruments.len() - 1
                                }
                            };
                            instruments[index].notes.push(Note {
                                velocity: *velocity,
                                pitch,
                                start: time(*start),
                                end: time(end),
                            });
                        }
                        if !close.is_empty() && !keep.is_empty() {
                            open[slot].1 = keep;
                        } else {
                            open.remove(slot);
                        }
                    }
                    _ => {}
                }
            }
        }
        let max_tick = raw_tracks
            .iter()
            .flat_map(|t| t.iter().map(|e| e.tick))
            .max()
            .unwrap_or(0);
        Ok(Self {
            resolution,
            tempo_us,
            instruments,
            tick_table_len: max_tick + 2,
        })
    }
}

/// Open notes per `(channel, pitch)`: `(note-on tick, velocity)` in arrival order.
type OpenNotes = Vec<((u8, u8), Vec<(u64, u8)>)>;

fn push_varlen(out: &mut Vec<u8>, mut value: u64) {
    let mut stack = vec![(value & 0x7F) as u8];
    value >>= 7;
    while value > 0 {
        stack.push(((value & 0x7F) as u8) | 0x80);
        value >>= 7;
    }
    out.extend(stack.into_iter().rev());
}

fn push_delta(out: &mut Vec<u8>, delta: u64) {
    push_varlen(out, delta);
}

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], Error> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&e| e <= self.bytes.len())
            .ok_or_else(|| Error::Symbolic("malformed MIDI: truncated".into()))?;
        let slice = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(slice)
    }

    fn u8(&mut self) -> Result<u8, Error> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32, Error> {
        let b = self.take(4)?;
        Ok(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn varlen(&mut self) -> Result<u64, Error> {
        let mut value = 0u64;
        for _ in 0..4 {
            let b = self.u8()?;
            value = (value << 7) | u64::from(b & 0x7F);
            if b & 0x80 == 0 {
                return Ok(value);
            }
        }
        Err(Error::Symbolic("malformed MIDI: varlen too long".into()))
    }
}

#[derive(Clone, Debug, PartialEq)]
enum TrackEvent {
    Tempo(u32),
    Name(String),
    Program(u8, u8),
    NoteOn(u8, u8, u8),
    NoteOff(u8, u8),
    Other,
}

struct TimedEvent {
    tick: u64,
    kind: TrackEvent,
}

fn parse_track(bytes: &[u8]) -> Result<Vec<TimedEvent>, Error> {
    let mut reader = Reader { bytes, pos: 0 };
    let mut tick = 0u64;
    let mut running: Option<u8> = None;
    let mut events = Vec::new();
    while reader.pos < bytes.len() {
        tick += reader.varlen()?;
        let mut status = reader.u8()?;
        if status == 0xFF {
            let kind = reader.u8()?;
            let len = reader.varlen()? as usize;
            let data = reader.take(len)?;
            running = None;
            let event = match kind {
                0x51 if len == 3 => {
                    TrackEvent::Tempo(u32::from_be_bytes([0, data[0], data[1], data[2]]))
                }
                0x03 => TrackEvent::Name(data.iter().map(|&b| b as char).collect()),
                _ => TrackEvent::Other,
            };
            events.push(TimedEvent { tick, kind: event });
            continue;
        }
        if status == 0xF0 || status == 0xF7 {
            let len = reader.varlen()? as usize;
            reader.take(len)?;
            running = None;
            continue;
        }
        let first_data = if status < 0x80 {
            let data = status;
            status = running.ok_or_else(|| {
                Error::Symbolic("malformed MIDI: running status without status".into())
            })?;
            Some(data)
        } else {
            running = Some(status);
            None
        };
        let next = |reader: &mut Reader| -> Result<u8, Error> {
            match first_data {
                Some(d) => Ok(d),
                None => reader.u8(),
            }
        };
        let channel = status & 0x0F;
        let event = match status & 0xF0 {
            0x90 => {
                let pitch = next(&mut reader)?;
                TrackEvent::NoteOn(channel, pitch, reader.u8()?)
            }
            0x80 => {
                let pitch = next(&mut reader)?;
                reader.u8()?;
                TrackEvent::NoteOff(channel, pitch)
            }
            0xC0 => TrackEvent::Program(channel, next(&mut reader)?),
            0xD0 => {
                next(&mut reader)?;
                TrackEvent::Other
            }
            _ => {
                next(&mut reader)?;
                reader.u8()?;
                TrackEvent::Other
            }
        };
        events.push(TimedEvent { tick, kind: event });
    }
    Ok(events)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Writing the notes read back from an upstream file reproduces it byte for byte (committed
    /// `real_full` files written by pretty_midi 0.2.10 / mido 1.3.3; every instrument in them has
    /// notes, so a load keeps them all).
    ///
    /// Mutation that must fail: drop running status, or swap the note-on/note-off sort order.
    #[test]
    fn round_trips_an_upstream_file_byte_for_byte() {
        for bytes in [
            &include_bytes!(
                "../../../../scripts/reference/sheetsage2/artifacts/real_full/transcription.mid"
            )[..],
            &include_bytes!(
                "../../../../scripts/reference/sheetsage2/artifacts/real_full/chords.mid"
            )[..],
        ] {
            let midi = Midi::from_bytes(bytes).unwrap();
            assert_eq!(midi.to_bytes(), bytes);
        }
    }

    /// A fresh object rounds times to ticks half to even (`int(round(numpy.float64))`), so exact
    /// half-ticks go to the even tick. Mutation that must fail: `round()` (half away from zero).
    #[test]
    fn exact_half_ticks_round_to_even() {
        let scale = tick_scale(960, None);
        let mut midi = Midi::new(960);
        midi.instruments.push(Instrument {
            program: 0,
            name: "Vocal".into(),
            notes: vec![Note {
                velocity: 100,
                pitch: 60,
                start: 2.5 * scale,
                end: 5.5 * scale,
            }],
        });
        assert_eq!(
            (2.5 * scale) / scale,
            2.5,
            "the probe must sit exactly on a half tick"
        );
        let back = Midi::from_bytes(&midi.to_bytes()).unwrap();
        let note = back.instruments[0].notes[0];
        assert_eq!((note.start / scale).round(), 2.0);
        assert_eq!((note.end / scale).round(), 6.0);
    }

    #[test]
    fn a_note_shorter_than_half_a_tick_is_dropped_like_pretty_midi() {
        let mut midi = Midi::new(960);
        midi.instruments.push(Instrument {
            program: 0,
            name: "Vocal".into(),
            notes: vec![
                Note {
                    velocity: 100,
                    pitch: 60,
                    start: 0.0,
                    end: 0.5,
                },
                Note {
                    velocity: 100,
                    pitch: 62,
                    start: 1.0,
                    end: 1.0001,
                },
            ],
        });
        let back = Midi::from_bytes(&midi.to_bytes()).unwrap();
        assert_eq!(back.instruments.len(), 1);
        assert_eq!(back.instruments[0].notes.len(), 1);
        assert_eq!(
            back.instruments[0].notes[0].end,
            960.0 * tick_scale(960, None)
        );
    }
}
