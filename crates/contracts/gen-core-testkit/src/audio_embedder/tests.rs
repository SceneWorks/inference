//! The audio-embedder testkit verifying itself: a configurable in-crate stub drives each check, and
//! one deliberately-broken variant per check proves the check fires (sc-12853). Pure-host — runs on
//! the Linux gen-core lane.

use super::*;
use gen_core::registry::AudioEmbedderRegistration;
use gen_core::runtime::LoadSpec;
use gen_core::{AudioEmbedder, AudioEmbedderDescriptor};

const STUB_ID: &str = "testkit_audio_embedder_stub";
const UNREG_ID: &str = "testkit_audio_embedder_unregistered_stub";
const DIM: usize = 512;

/// Which contract guarantees the stub upholds.
#[derive(Clone, Copy)]
struct Behavior {
    /// Both vectors are L2-normalized (vs. left un-normalized).
    normalized: bool,
    /// The text vector matches the audio vector's dimension (vs. one element short).
    text_dim_matches: bool,
    /// Vectors are finite (vs. a NaN).
    finite: bool,
}

impl Behavior {
    fn good() -> Self {
        Self {
            normalized: true,
            text_dim_matches: true,
            finite: true,
        }
    }
}

struct StubAudioEmbedder {
    desc: AudioEmbedderDescriptor,
    behavior: Behavior,
}

fn stub_desc(id: &'static str) -> AudioEmbedderDescriptor {
    AudioEmbedderDescriptor {
        id,
        family: "audio-embed",
        backend: "stub",
        embedding_dim: DIM,
        space: "testkit-space",
        mac_only: false,
    }
}

/// A unit vector of `len` elements (each `1/sqrt(len)`), optionally left un-normalized or salted with
/// a NaN.
fn vector(len: usize, normalized: bool, finite: bool) -> Vec<f32> {
    let value = if normalized {
        1.0 / (len as f32).sqrt()
    } else {
        1.0 // ‖v‖ = sqrt(len) ≫ 1
    };
    let mut v = vec![value; len];
    if !finite {
        v[0] = f32::NAN;
    }
    v
}

impl StubAudioEmbedder {
    fn new(id: &'static str, behavior: Behavior) -> Self {
        Self {
            desc: stub_desc(id),
            behavior,
        }
    }
    fn boxed(id: &'static str, behavior: Behavior) -> Box<dyn AudioEmbedder> {
        Box::new(Self::new(id, behavior))
    }
}

impl AudioEmbedder for StubAudioEmbedder {
    fn descriptor(&self) -> &AudioEmbedderDescriptor {
        &self.desc
    }
    fn embed(&self, _audio: &gen_core::AudioTrack) -> gen_core::Result<Vec<f32>> {
        Ok(vector(DIM, self.behavior.normalized, self.behavior.finite))
    }
    fn embed_text(&self, _text: &str) -> gen_core::Result<Vec<f32>> {
        let len = if self.behavior.text_dim_matches {
            DIM
        } else {
            DIM - 1
        };
        // Keep the text vector finite/normalized per its own length so only the tested axis breaks.
        Ok(vector(len, self.behavior.normalized, true))
    }
}

fn stub_descriptor() -> AudioEmbedderDescriptor {
    stub_desc(STUB_ID)
}
fn stub_load(_spec: &LoadSpec) -> gen_core::Result<Box<dyn AudioEmbedder>> {
    Ok(StubAudioEmbedder::boxed(STUB_ID, Behavior::good()))
}
const STUB_REGISTRATION: AudioEmbedderRegistration = AudioEmbedderRegistration {
    descriptor: stub_descriptor,
    load: stub_load,
};

fn registry() -> gen_core::ProviderRegistry {
    gen_core::ProviderRegistryBuilder::new()
        .register_audio_embedder(STUB_REGISTRATION)
        .build()
        .expect("stub audio-embedder registry should build")
}

fn cheap() -> AudioEmbedderProfile {
    AudioEmbedderProfile::cheap()
}

#[test]
fn good_stub_passes_full_conformance() {
    audio_embedder_conformance(
        || StubAudioEmbedder::boxed(STUB_ID, Behavior::good()),
        &cheap(),
    );
}

#[test]
fn good_stub_passes_every_check_individually() {
    let e = StubAudioEmbedder::new(STUB_ID, Behavior::good());
    check_audio_embed_joint(&e, &cheap()).unwrap();
    check_audio_embedder_registry(&registry(), &e).unwrap();
}

#[test]
fn unnormalized_fails_joint_check() {
    let e = StubAudioEmbedder::new(
        STUB_ID,
        Behavior {
            normalized: false,
            ..Behavior::good()
        },
    );
    let err = check_audio_embed_joint(&e, &cheap()).unwrap_err();
    assert!(err.contains("L2 norm"), "got: {err}");
}

#[test]
fn mismatched_dim_fails_joint_check() {
    let e = StubAudioEmbedder::new(
        STUB_ID,
        Behavior {
            text_dim_matches: false,
            ..Behavior::good()
        },
    );
    let err = check_audio_embed_joint(&e, &cheap()).unwrap_err();
    // Caught by the per-vector dim check (text vector is DIM-1 vs advertised DIM).
    assert!(err.contains("advertises embedding_dim"), "got: {err}");
}

#[test]
fn non_finite_fails_joint_check() {
    let e = StubAudioEmbedder::new(
        STUB_ID,
        Behavior {
            finite: false,
            ..Behavior::good()
        },
    );
    let err = check_audio_embed_joint(&e, &cheap()).unwrap_err();
    assert!(err.contains("non-finite"), "got: {err}");
}

#[test]
fn unregistered_id_fails_registry_check() {
    let e = StubAudioEmbedder::new(UNREG_ID, Behavior::good());
    assert!(check_audio_embedder_registry(&registry(), &e).is_err());
}

#[test]
#[should_panic(expected = "audio-embedder conformance FAILED")]
fn conformance_panics_on_a_broken_stub() {
    audio_embedder_conformance(
        || {
            StubAudioEmbedder::boxed(
                STUB_ID,
                Behavior {
                    normalized: false,
                    ..Behavior::good()
                },
            )
        },
        &cheap(),
    );
}
