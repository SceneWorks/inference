//! Receipt scopes follow each model through both eager construction and deferred execution.
use super::*;
use crate::caption::{CaptionOutput, CaptionRequest};
use crate::media::{AudioTrack, Image};
use crate::runtime::Progress;
use crate::transform::{CompletedTransform, TransformRequest};
use crate::{
    AudioTransformRequest, CompletedAudioTransform, TranscribeRequest, TranscriptOutput,
    VoiceEmbedding,
};
use crate::{CheckpointReceiptScope, CheckpointWeightFacts};

struct Scoped<T: ?Sized> {
    inner: Box<T>,
    scope: CheckpointReceiptScope,
}
macro_rules! scope_model {
    ($factory:ident, $model:ident, $desc:ty; $(fn $method:ident($($arg:ident: $ty:ty),*) -> $ret:ty;)*) => {
        pub(super) fn $factory(spec: &LoadSpec, load: impl FnOnce()->Result<Box<dyn $model>>) -> Result<Box<dyn $model>> {
            let scope=CheckpointReceiptScope::for_load(spec)?;
            let inner=scope.with(load)?;
            Ok(Box::new(Scoped {inner,scope}))
        }
        impl $model for Scoped<dyn $model> {
            fn descriptor(&self)->&$desc{self.inner.descriptor()}
            fn checkpoint_weight_facts(&self)->Option<CheckpointWeightFacts>{
                self.scope.facts().or_else(||self.inner.checkpoint_weight_facts())
            }
            fn checkpoint_weight_facts_by_source(&self)->Vec<CheckpointWeightFacts>{
                let facts=self.scope.snapshots();
                if facts.is_empty(){self.inner.checkpoint_weight_facts_by_source()}else{facts}
            }
            $(fn $method(&self,$($arg:$ty),*) -> $ret {
                self.scope.with_prepared(||self.inner.$method($($arg),*))
            })*
        }
    }
}
scope_model!(load_transform,Transform,TransformDescriptor;
    fn validate(req:&TransformRequest)->Result<()>;
    fn apply(req:&TransformRequest,on_progress:&mut dyn FnMut(Progress))->Result<Image>;
    fn apply_with_execution_report(req:&TransformRequest,on_progress:&mut dyn FnMut(Progress))->Result<CompletedTransform>;
);
scope_model!(load_captioner,Captioner,CaptionerDescriptor;
    fn validate(req:&CaptionRequest)->Result<()>;
    fn caption(req:&CaptionRequest,on_progress:&mut dyn FnMut(Progress))->Result<CaptionOutput>;
);
scope_model!(load_image_embedder,ImageEmbedder,ImageEmbedderDescriptor;
    fn embed(image:&Image)->Result<Vec<f32>>;
    fn embed_batch(images:&[Image])->Result<Vec<Vec<f32>>>;
);
scope_model!(load_text_embedder,TextEmbedder,TextEmbedderDescriptor;
    fn embed_text(text:&str)->Result<Vec<f32>>;
    fn embed_text_batch(texts:&[&str])->Result<Vec<Vec<f32>>>;
);

scope_model!(load_audio_embedder,AudioEmbedder,AudioEmbedderDescriptor;
    fn embed(audio: &AudioTrack) -> Result<Vec<f32>>;
    fn embed_text(text: &str) -> Result<Vec<f32>>;
    fn embed_batch(audio: &[AudioTrack]) -> Result<Vec<Vec<f32>>>;
    fn embed_text_batch(texts: &[&str]) -> Result<Vec<Vec<f32>>>;
);
scope_model!(load_voice_embedder,VoiceEmbedder,VoiceEmbedderDescriptor;
    fn embed(audio: &AudioTrack) -> Result<VoiceEmbedding>;
);
scope_model!(load_transcriber,Transcriber,TranscriberDescriptor;
    fn validate(req: &TranscribeRequest) -> Result<()>;
    fn transcribe(req: &TranscribeRequest, progress: &mut dyn FnMut(Progress)) -> Result<TranscriptOutput>;
);
scope_model!(load_audio_transform,AudioTransform,AudioTransformDescriptor;
    fn validate(req: &AudioTransformRequest) -> Result<()>;
    fn apply(req: &AudioTransformRequest, progress: &mut dyn FnMut(Progress)) -> Result<Vec<AudioTrack>>;
    fn apply_with_execution_report(req: &AudioTransformRequest, progress: &mut dyn FnMut(Progress)) -> Result<CompletedAudioTransform>;
);
