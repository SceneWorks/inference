//! Capture checkpoints opened while loading or executing one registered audio model.
use crate::{
    AudioEmbedder, AudioEmbedderDescriptor, AudioTrack, AudioTransform, AudioTransformDescriptor,
    AudioTransformRequest, CheckpointReceiptScope, CheckpointWeightFacts, CompletedAudioTransform,
    LoadSpec, Progress, Result, TranscribeRequest, Transcriber, TranscriberDescriptor,
    TranscriptOutput, VoiceEmbedder, VoiceEmbedderDescriptor, VoiceEmbedding,
};

struct ScopedAudio<T: ?Sized> {
    inner: Box<T>,
    scope: CheckpointReceiptScope,
}

macro_rules! loader {
    ($name:ident, $provider:ident) => {
        pub(super) fn $name(
            spec: &LoadSpec,
            load: impl FnOnce() -> Result<Box<dyn $provider>>,
        ) -> Result<Box<dyn $provider>> {
            let scope = CheckpointReceiptScope::for_load(spec)?;
            let inner = scope.with(load)?;
            Ok(Box::new(ScopedAudio { inner, scope }))
        }
    };
}
loader!(load_audio_embedder, AudioEmbedder);
loader!(load_audio_transform, AudioTransform);
loader!(load_transcriber, Transcriber);
loader!(load_voice_embedder, VoiceEmbedder);

macro_rules! receipts {
    () => {
        fn checkpoint_weight_facts(&self) -> Option<CheckpointWeightFacts> {
            self.inner
                .checkpoint_weight_facts()
                .or_else(|| self.scope.facts())
        }
        fn checkpoint_weight_facts_by_source(&self) -> Vec<CheckpointWeightFacts> {
            let mut facts = self.scope.snapshots();
            for inner in self.inner.checkpoint_weight_facts_by_source() {
                if !facts.contains(&inner) {
                    facts.push(inner);
                }
            }
            facts
        }
    };
}

impl AudioEmbedder for ScopedAudio<dyn AudioEmbedder> {
    receipts!();
    fn descriptor(&self) -> &AudioEmbedderDescriptor {
        self.inner.descriptor()
    }
    fn embed(&self, audio: &AudioTrack) -> Result<Vec<f32>> {
        self.scope.with(|| self.inner.embed(audio))
    }
    fn embed_text(&self, text: &str) -> Result<Vec<f32>> {
        self.scope.with(|| self.inner.embed_text(text))
    }
    fn embed_batch(&self, audio: &[AudioTrack]) -> Result<Vec<Vec<f32>>> {
        self.scope.with(|| self.inner.embed_batch(audio))
    }
    fn embed_text_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        self.scope.with(|| self.inner.embed_text_batch(texts))
    }
}

impl VoiceEmbedder for ScopedAudio<dyn VoiceEmbedder> {
    receipts!();
    fn descriptor(&self) -> &VoiceEmbedderDescriptor {
        self.inner.descriptor()
    }
    fn embed(&self, audio: &AudioTrack) -> Result<VoiceEmbedding> {
        self.scope.with(|| self.inner.embed(audio))
    }
}

impl Transcriber for ScopedAudio<dyn Transcriber> {
    receipts!();
    fn descriptor(&self) -> &TranscriberDescriptor {
        self.inner.descriptor()
    }
    fn validate(&self, req: &TranscribeRequest) -> Result<()> {
        self.inner.validate(req)
    }
    fn transcribe(
        &self,
        req: &TranscribeRequest,
        progress: &mut dyn FnMut(Progress),
    ) -> Result<TranscriptOutput> {
        self.scope.with(|| self.inner.transcribe(req, progress))
    }
}

impl AudioTransform for ScopedAudio<dyn AudioTransform> {
    receipts!();
    fn descriptor(&self) -> &AudioTransformDescriptor {
        self.inner.descriptor()
    }
    fn validate(&self, req: &AudioTransformRequest) -> Result<()> {
        self.inner.validate(req)
    }
    fn apply(
        &self,
        req: &AudioTransformRequest,
        progress: &mut dyn FnMut(Progress),
    ) -> Result<Vec<AudioTrack>> {
        self.scope.with(|| self.inner.apply(req, progress))
    }
    fn apply_with_execution_report(
        &self,
        req: &AudioTransformRequest,
        progress: &mut dyn FnMut(Progress),
    ) -> Result<CompletedAudioTransform> {
        self.scope
            .with(|| self.inner.apply_with_execution_report(req, progress))
    }
}
