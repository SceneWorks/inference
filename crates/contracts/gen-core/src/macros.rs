/// Define a generator registration constant for an explicit provider catalog.
///
/// The optional `footprint` form keeps provider-owned component accounting attached to the same
/// explicit registration that owns loading; no process-global inventory is involved.
#[macro_export]
macro_rules! register_generators {
    ( $vis:vis const $name:ident = $desc:path => $load:path $(,)? ) => {
        $vis const $name: $crate::registry::ModelRegistration =
            $crate::registry::ModelRegistration {
                descriptor: $desc,
                load: |spec| $load(spec).map_err(::core::convert::Into::into),
                footprint: ::core::option::Option::None,
            };
    };
    ( $vis:vis const $name:ident = $desc:path => $load:path ; footprint = $fp:path $(,)? ) => {
        $vis const $name: $crate::registry::ModelRegistration =
            $crate::registry::ModelRegistration {
                descriptor: $desc,
                load: |spec| $load(spec).map_err(::core::convert::Into::into),
                footprint: ::core::option::Option::Some(
                    |spec| $fp(spec).map_err(::core::convert::Into::into)
                ),
            };
    };
}

/// Define a trainer registration constant for an explicit provider catalog.
#[macro_export]
macro_rules! register_trainer {
    ( $vis:vis const $name:ident = $desc:path => $load:path $(,)? ) => {
        $vis const $name: $crate::registry::TrainerRegistration =
            $crate::registry::TrainerRegistration {
                descriptor: $desc,
                load: |spec| $load(spec).map_err(::core::convert::Into::into),
            };
    };
}

/// Define a captioner registration constant for an explicit provider catalog.
#[macro_export]
macro_rules! register_captioner {
    ( $vis:vis const $name:ident = $desc:path => $load:path $(,)? ) => {
        $vis const $name: $crate::registry::CaptionerRegistration =
            $crate::registry::CaptionerRegistration {
                descriptor: $desc,
                load: |spec| $load(spec).map_err(::core::convert::Into::into),
            };
    };
}

/// Define a transcriber registration constant for an explicit provider catalog (sc-12850) —
/// the audio-to-text sibling of [`register_captioner`](crate::register_captioner).
#[macro_export]
macro_rules! register_transcriber {
    ( $vis:vis const $name:ident = $desc:path => $load:path $(,)? ) => {
        $vis const $name: $crate::registry::TranscriberRegistration =
            $crate::registry::TranscriberRegistration {
                descriptor: $desc,
                load: |spec| $load(spec).map_err(::core::convert::Into::into),
            };
    };
}

/// Define an image-embedder registration constant for an explicit provider catalog.
#[macro_export]
macro_rules! register_image_embedder {
    ( $vis:vis const $name:ident = $desc:path => $load:path $(,)? ) => {
        $vis const $name: $crate::registry::ImageEmbedderRegistration =
            $crate::registry::ImageEmbedderRegistration {
                descriptor: $desc,
                load: |spec| $load(spec).map_err(::core::convert::Into::into),
            };
    };
}

/// Define a text-embedder registration constant for an explicit provider catalog.
#[macro_export]
macro_rules! register_text_embedder {
    ( $vis:vis const $name:ident = $desc:path => $load:path $(,)? ) => {
        $vis const $name: $crate::registry::TextEmbedderRegistration =
            $crate::registry::TextEmbedderRegistration {
                descriptor: $desc,
                load: |spec| $load(spec).map_err(::core::convert::Into::into),
            };
    };
}

/// Define a voice-embedder registration constant for an explicit provider catalog.
#[macro_export]
macro_rules! register_voice_embedder {
    ( $vis:vis const $name:ident = $desc:path => $load:path $(,)? ) => {
        $vis const $name: $crate::registry::VoiceEmbedderRegistration =
            $crate::registry::VoiceEmbedderRegistration {
                descriptor: $desc,
                load: |spec| $load(spec).map_err(::core::convert::Into::into),
            };
    };
}

/// Define an audio-embedder registration constant for an explicit provider catalog (sc-12851) —
/// the semantic audio-text (CLAP-style) sibling of
/// [`register_image_embedder`](crate::register_image_embedder).
#[macro_export]
macro_rules! register_audio_embedder {
    ( $vis:vis const $name:ident = $desc:path => $load:path $(,)? ) => {
        $vis const $name: $crate::registry::AudioEmbedderRegistration =
            $crate::registry::AudioEmbedderRegistration {
                descriptor: $desc,
                load: |spec| $load(spec).map_err(::core::convert::Into::into),
            };
    };
}

/// Define an audio-transform registration constant for an explicit provider catalog (sc-12839),
/// parallel to [`register_voice_embedder`](crate::register_voice_embedder).
#[macro_export]
macro_rules! register_audio_transform {
    ( $vis:vis const $name:ident = $desc:path => $load:path $(,)? ) => {
        $vis const $name: $crate::registry::AudioTransformRegistration =
            $crate::registry::AudioTransformRegistration {
                descriptor: $desc,
                load: |spec| $load(spec).map_err(::core::convert::Into::into),
            };
    };
}

/// Implement the standard delegation-pattern [`crate::Generator`] wrapper for provider structs.
#[macro_export]
macro_rules! impl_generator {
    (
        $ty:ty {
            validate: |$self_arg:ident, $req_arg:ident| $validate:expr,
            generate: $generate:ident $(,)?
        }
    ) => {
        impl $crate::Generator for $ty {
            fn descriptor(&self) -> &$crate::ModelDescriptor {
                &self.descriptor
            }

            fn validate(&self, req: &$crate::GenerationRequest) -> $crate::Result<()> {
                let validate = |$self_arg: &Self, $req_arg: &$crate::GenerationRequest| $validate;
                validate(self, req).map_err(::core::convert::Into::into)
            }

            fn generate(
                &self,
                req: &$crate::GenerationRequest,
                on_progress: &mut dyn FnMut($crate::Progress),
            ) -> $crate::Result<$crate::GenerationOutput> {
                self.$generate(req, on_progress)
                    .map_err(::core::convert::Into::into)
            }
        }
    };
}
