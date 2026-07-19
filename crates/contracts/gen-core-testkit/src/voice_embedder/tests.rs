//! The voice-embedder testkit verifying itself: a configurable in-crate stub drives each check, and
//! one deliberately-broken variant per check proves the check fires (sc-12853). Pure-host — runs on
//! the Linux gen-core lane.

use super::*;
use gen_core::registry::VoiceEmbedderRegistration;
use gen_core::runtime::LoadSpec;
use gen_core::{VoiceEmbedder, VoiceEmbedderDescriptor, VoiceEmbedding};

const STUB_ID: &str = "testkit_voice_embedder_stub";
const UNREG_ID: &str = "testkit_voice_embedder_unregistered_stub";
const DIM: usize = 256;

/// Which contract guarantees the stub upholds.
#[derive(Clone, Copy)]
struct Behavior {
    /// Returns exactly `embedding_dim` elements (vs. one too many).
    correct_dim: bool,
    /// Returns finite values (vs. a NaN).
    finite: bool,
    /// Rejects a too-short / empty clip (vs. embedding it anyway).
    reject_short: bool,
}

impl Behavior {
    fn good() -> Self {
        Self {
            correct_dim: true,
            finite: true,
            reject_short: true,
        }
    }
}

struct StubVoiceEmbedder {
    desc: VoiceEmbedderDescriptor,
    behavior: Behavior,
}

fn stub_desc(id: &'static str) -> VoiceEmbedderDescriptor {
    VoiceEmbedderDescriptor {
        id,
        family: "voice",
        backend: "stub",
        embedding_dim: DIM,
        mac_only: false,
    }
}

impl StubVoiceEmbedder {
    fn new(id: &'static str, behavior: Behavior) -> Self {
        Self {
            desc: stub_desc(id),
            behavior,
        }
    }
    fn boxed(id: &'static str, behavior: Behavior) -> Box<dyn VoiceEmbedder> {
        Box::new(Self::new(id, behavior))
    }
}

impl VoiceEmbedder for StubVoiceEmbedder {
    fn descriptor(&self) -> &VoiceEmbedderDescriptor {
        &self.desc
    }
    fn embed(&self, audio: &gen_core::AudioTrack) -> gen_core::Result<VoiceEmbedding> {
        if self.behavior.reject_short && audio.samples.len() < 16 {
            return Err(gen_core::Error::Msg(
                "stub voice embedder: reference clip too short".into(),
            ));
        }
        let len = if self.behavior.correct_dim {
            DIM
        } else {
            DIM + 1
        };
        let mut v = vec![audio.samples.len() as f32; len];
        if !self.behavior.finite {
            v[0] = f32::NAN;
        }
        Ok(v)
    }
}

fn stub_descriptor() -> VoiceEmbedderDescriptor {
    stub_desc(STUB_ID)
}
fn stub_load(_spec: &LoadSpec) -> gen_core::Result<Box<dyn VoiceEmbedder>> {
    Ok(StubVoiceEmbedder::boxed(STUB_ID, Behavior::good()))
}
const STUB_REGISTRATION: VoiceEmbedderRegistration = VoiceEmbedderRegistration {
    descriptor: stub_descriptor,
    load: stub_load,
};

fn registry() -> gen_core::ProviderRegistry {
    gen_core::ProviderRegistryBuilder::new()
        .register_voice_embedder(STUB_REGISTRATION)
        .build()
        .expect("stub voice-embedder registry should build")
}

fn cheap() -> VoiceEmbedderProfile {
    VoiceEmbedderProfile::cheap()
}

#[test]
fn good_stub_passes_full_conformance() {
    voice_embedder_conformance(
        || StubVoiceEmbedder::boxed(STUB_ID, Behavior::good()),
        &cheap(),
    );
}

#[test]
fn good_stub_passes_every_check_individually() {
    let e = StubVoiceEmbedder::new(STUB_ID, Behavior::good());
    check_voice_embed(&e, &cheap()).unwrap();
    check_voice_embed_rejects_short(&e, &cheap()).unwrap();
    check_voice_embedder_registry(&registry(), &e).unwrap();
}

#[test]
fn wrong_dim_fails_embed_check() {
    let e = StubVoiceEmbedder::new(
        STUB_ID,
        Behavior {
            correct_dim: false,
            ..Behavior::good()
        },
    );
    let err = check_voice_embed(&e, &cheap()).unwrap_err();
    assert!(err.contains("advertises embedding_dim"), "got: {err}");
}

#[test]
fn non_finite_fails_embed_check() {
    let e = StubVoiceEmbedder::new(
        STUB_ID,
        Behavior {
            finite: false,
            ..Behavior::good()
        },
    );
    let err = check_voice_embed(&e, &cheap()).unwrap_err();
    assert!(err.contains("non-finite"), "got: {err}");
}

#[test]
fn accepting_short_clip_fails_rejection_check() {
    let e = StubVoiceEmbedder::new(
        STUB_ID,
        Behavior {
            reject_short: false,
            ..Behavior::good()
        },
    );
    let err = check_voice_embed_rejects_short(&e, &cheap()).unwrap_err();
    assert!(err.contains("must reject too-short input"), "got: {err}");
}

#[test]
fn unregistered_id_fails_registry_check() {
    let e = StubVoiceEmbedder::new(UNREG_ID, Behavior::good());
    assert!(check_voice_embedder_registry(&registry(), &e).is_err());
}

#[test]
#[should_panic(expected = "voice-embedder conformance FAILED")]
fn conformance_panics_on_a_broken_stub() {
    voice_embedder_conformance(
        || {
            StubVoiceEmbedder::boxed(
                STUB_ID,
                Behavior {
                    correct_dim: false,
                    ..Behavior::good()
                },
            )
        },
        &cheap(),
    );
}
