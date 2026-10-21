//! Borrowed streaming exact work binding. No deep request/pixel/prompt clone or whole Debug String.
//! The explicit field projection mirrors derived Debug with only seed/cancel neutralized; a tiny
//! oracle test detects any newly added request field omitted from this view.
use std::fmt::{self, Write};
use std::sync::OnceLock;

use mlx_gen::gen_core::iris::depth::DepthRequest;
use mlx_gen::gen_core::{CancelFlag, GenerationRequest, TransformRequest};
use sha2::{Digest, Sha256};

use crate::ios_admission::IosWork;

const DOMAIN: &[u8] = b"ios-exact-work-binding-v2\0";

fn neutral_cancel() -> &'static CancelFlag {
    static CANCEL: OnceLock<CancelFlag> = OnceLock::new();
    CANCEL.get_or_init(CancelFlag::default)
}

struct DigestWriter<'a>(&'a mut Sha256);
impl fmt::Write for DigestWriter<'_> {
    fn write_str(&mut self, text: &str) -> fmt::Result {
        self.0.update(text.as_bytes());
        Ok(())
    }
}

struct GenerationView<'a>(&'a GenerationRequest);
impl fmt::Debug for GenerationView<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let r = self.0;
        f.debug_struct("GenerationRequest")
            .field("prompt", &r.prompt)
            .field("negative_prompt", &r.negative_prompt)
            .field("width", &r.width)
            .field("height", &r.height)
            .field("count", &r.count)
            .field("seed", &Option::<u64>::None)
            .field("steps", &r.steps)
            .field("guidance", &r.guidance)
            .field("true_cfg", &r.true_cfg)
            .field("timestep_to_start_cfg", &r.timestep_to_start_cfg)
            .field("sampler", &r.sampler)
            .field("scheduler", &r.scheduler)
            .field("scheduler_shift", &r.scheduler_shift)
            .field("guidance_method", &r.guidance_method)
            .field("guidance_eta", &r.guidance_eta)
            .field("guidance_momentum", &r.guidance_momentum)
            .field("guidance_norm_threshold", &r.guidance_norm_threshold)
            .field("cfg_interval", &r.cfg_interval)
            .field("prompt_batch", &r.prompt_batch)
            .field("caption_overflow", &r.caption_overflow)
            .field("conditioning", &r.conditioning)
            .field("strength", &r.strength)
            .field("control_scale", &r.control_scale)
            .field("text_style_gain", &r.text_style_gain)
            .field("image_guidance", &r.image_guidance)
            .field("frames", &r.frames)
            .field("fps", &r.fps)
            .field("duration", &r.duration)
            .field("video_mode", &r.video_mode)
            .field("veda", &r.veda)
            .field("trim_first_frames", &r.trim_first_frames)
            .field("num_generated_keyframes", &r.num_generated_keyframes)
            .field("reference_image_short_edge", &r.reference_image_short_edge)
            .field("temporal_upsample_rounds", &r.temporal_upsample_rounds)
            .field("auto_duration", &r.auto_duration)
            .field("motion_bucket_id", &r.motion_bucket_id)
            .field("noise_aug_strength", &r.noise_aug_strength)
            .field("decode_chunk_size", &r.decode_chunk_size)
            .field("conditioning_fps", &r.conditioning_fps)
            .field("softness", &r.softness)
            .field("enhance_prompt", &r.enhance_prompt)
            .field("use_uncensored_enhancer", &r.use_uncensored_enhancer)
            .field("enhance_max_tokens", &r.enhance_max_tokens)
            .field("enhance_temperature", &r.enhance_temperature)
            .field("prompt_enhancement", &r.prompt_enhancement)
            .field("use_pid", &r.use_pid)
            .field("pid_capture_sigma", &r.pid_capture_sigma)
            .field("memory", &r.memory)
            .field("approximation", &r.approximation)
            .field("audio", &r.audio)
            .field("phases", &r.phases)
            .field("cancel", neutral_cancel())
            .field("preview", &r.preview)
            .field("hdr", &r.hdr)
            .field("output_channels", &r.output_channels)
            .finish()
    }
}

struct TransformView<'a>(&'a TransformRequest);
impl fmt::Debug for TransformView<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let r = self.0;
        f.debug_struct("TransformRequest")
            .field("image", &r.image)
            .field("target", &r.target)
            .field("seed", &Option::<u64>::None)
            .field("strength", &r.strength)
            .field("steps", &r.steps)
            .field("input_sizing", &r.input_sizing)
            .field("color_fix", &r.color_fix)
            .field("cancel", neutral_cancel())
            .finish()
    }
}

struct DepthView<'a>(&'a DepthRequest);
impl fmt::Debug for DepthView<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DepthRequest")
            .field("image", &self.0.image)
            .field("resolution", &self.0.resolution)
            .field("cancel", neutral_cancel())
            .finish()
    }
}

/// Bind a borrowed generation request with the existing exact-work byte projection.
/// The owned-work entry below remains the independent compatibility oracle.
pub(crate) fn generation_binding(request: &GenerationRequest) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(DOMAIN);
    let mut sink = DigestWriter(&mut hash);
    write!(&mut sink, "generate:{:?}", GenerationView(request))
        .expect("Debug writes into an infallible SHA sink");
    hash.finalize().into()
}

pub(crate) fn work_binding(work: &IosWork) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(DOMAIN);
    let mut sink = DigestWriter(&mut hash);
    match work {
        IosWork::Generate(request) => write!(&mut sink, "generate:{:?}", GenerationView(request)),
        IosWork::Restore(request) => write!(&mut sink, "restore:{:?}", TransformView(request)),
        IosWork::Depth(request) => write!(&mut sink, "depth:{:?}", DepthView(request)),
    }
    .expect("Debug writes into an infallible SHA sink");
    hash.finalize().into()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use mlx_gen::gen_core::{Conditioning, Image};
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;

    thread_local! { static ALLOCATED: Cell<Option<usize>> = const { Cell::new(None) }; }
    struct CountingAllocator;
    fn record_allocation(bytes: usize) {
        let _ = ALLOCATED.try_with(|counter| {
            if let Some(current) = counter.get() {
                counter.set(Some(current + bytes));
            }
        });
    }
    // SAFETY: every allocation/deallocation delegates unchanged to System; the const TLS Cell
    // records only a scalar and performs no allocations or reentrant formatting.
    unsafe impl GlobalAlloc for CountingAllocator {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            record_allocation(layout.size());
            unsafe { System.alloc(layout) }
        }
        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            record_allocation(layout.size());
            unsafe { System.alloc_zeroed(layout) }
        }
        unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
            unsafe { System.dealloc(pointer, layout) }
        }
        unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, bytes: usize) -> *mut u8 {
            record_allocation(bytes);
            unsafe { System.realloc(pointer, layout, bytes) }
        }
    }
    #[global_allocator]
    static ALLOCATOR: CountingAllocator = CountingAllocator;

    pub(crate) fn allocations_during<T>(body: impl FnOnce() -> T) -> (T, usize) {
        struct Clear;
        impl Drop for Clear {
            fn drop(&mut self) {
                ALLOCATED.with(|counter| counter.set(None));
            }
        }
        ALLOCATED.with(|counter| {
            assert!(counter.get().is_none());
            counter.set(Some(0));
        });
        let clear = Clear;
        let value = body();
        let bytes =
            ALLOCATED.with(|counter| counter.get().expect("allocation scope remains active"));
        drop(clear);
        (value, bytes)
    }

    /// Tiny-only oracle. Whole formatting is intentionally forbidden in production and big-payload
    /// tests; here it detects a missing field if the public request struct grows.
    fn tiny_reference(work: &IosWork) -> [u8; 32] {
        let text = match work {
            IosWork::Generate(request) => {
                let mut r = request.clone();
                r.seed = None;
                r.cancel = CancelFlag::default();
                format!("generate:{r:?}")
            }
            IosWork::Restore(request) => {
                let mut r = request.clone();
                r.seed = None;
                r.cancel = CancelFlag::default();
                format!("restore:{r:?}")
            }
            IosWork::Depth(request) => {
                let mut r = request.clone();
                r.cancel = CancelFlag::default();
                format!("depth:{r:?}")
            }
        };
        let mut hash = Sha256::new();
        hash.update(DOMAIN);
        hash.update(text.as_bytes());
        hash.finalize().into()
    }

    #[test]
    fn borrowed_projection_covers_every_request_field_and_preserves_exclusions() {
        let mut generation = GenerationRequest {
            prompt: "binding λ\n".into(),
            ..Default::default()
        };
        generation.conditioning.push(Conditioning::Reference {
            image: Image {
                width: 2,
                height: 2,
                pixels: (0..12).collect(),
            },
            strength: Some(0.6),
        });
        let mut works = vec![
            IosWork::Generate(generation),
            IosWork::Restore(TransformRequest::default()),
            IosWork::Depth(DepthRequest::default()),
        ];
        for work in &works {
            assert_eq!(work_binding(work), tiny_reference(work));
        }
        for work in &mut works {
            let before = work_binding(work);
            match work {
                IosWork::Generate(r) => {
                    r.seed = Some(123);
                    r.cancel.cancel();
                }
                IosWork::Restore(r) => {
                    r.seed = Some(456);
                    r.cancel.cancel();
                }
                IosWork::Depth(r) => r.cancel.cancel(),
            }
            assert_eq!(
                work_binding(work),
                before,
                "only seed/cancellation are excluded"
            );
        }
        let IosWork::Generate(r) = &mut works[0] else {
            unreachable!()
        };
        r.scheduler = Some("karras".into());
        assert_ne!(
            work_binding(&works[0]),
            tiny_reference(&IosWork::Generate(GenerationRequest::default()))
        );
        assert_eq!(work_binding(&works[0]), tiny_reference(&works[0]));
    }

    #[test]
    fn big_borrowed_pixels_and_prompt_are_bound_without_mutating_or_copying_the_request() {
        let pixels: Vec<u8> = (0..(1024 * 512 * 3)).map(|i| (i % 251) as u8).collect();
        let mut request = GenerationRequest {
            prompt: "long borrowed prompt ".repeat(4096),
            conditioning: vec![Conditioning::Reference {
                image: Image {
                    width: 1024,
                    height: 512,
                    pixels,
                },
                strength: Some(0.6),
            }],
            ..Default::default()
        };
        let Conditioning::Reference { image, .. } = &request.conditioning[0] else {
            unreachable!()
        };
        let pointer = image.pixels.as_ptr();
        let capacity = image.pixels.capacity();
        let work = IosWork::Generate(request);
        // Initialize only fixed one-time state before measuring the pure borrowed operation.
        work_binding(&IosWork::Generate(GenerationRequest::default()));
        let (before, allocated) = allocations_during(|| work_binding(&work));
        assert_eq!(allocated, 0, "binding must not allocate payload-sized clones/Debug strings (or any Rust heap storage)");
        let IosWork::Generate(mut moved) = work else {
            unreachable!()
        };
        let Conditioning::Reference { image, .. } = &mut moved.conditioning[0] else {
            unreachable!()
        };
        assert_eq!(image.pixels.as_ptr(), pointer);
        assert_eq!(image.pixels.capacity(), capacity);
        image.pixels[17] ^= 1;
        request = moved;
        assert_ne!(before, work_binding(&IosWork::Generate(request)));
    }
}
