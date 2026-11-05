# Native CV ownership (SW-6775 / SW-6680)

`sceneworks-gen-cv` owns the existing SceneWorks native CV engines. It is composed
through the explicit `cv` feature of `runtime-macos` (with MLX) and `runtime-cuda` (ORT).
The LLM-only and iOS bundle graphs do not acquire this dependency.

| Capability | Inference owner | Preserved behavior |
| --- | --- | --- |
| Person detection | `person::Yolo`, `person::OrtYolo` | Entire YOLO11m MLX network including C3k/C3k2/SPPF/PSA/DFL, existing CPU-stream attention, ONNX execution, NHWC/NCHW letterbox, channel-major decode, class-0 filtering and greedy NMS |
| Whole-body pose | `pose::Detector` | YOLOX detection and RTMW SimCC, BGR affine sampling, original-pixel keypoints, body18/hands21/face68 mapping and square normalization |
| Super-resolution | `upscale::Upscaler` | x2/x4 RRDBNet ONNX execution, 512-pixel tiles, 16-pixel context, odd-edge replication, CHW conversion, validated output shape and RGB assembly, per-tile cancellation |
| ORT setup | `session` | CoreML default registration on Mac; CUDA error-on-failure off-Mac; one error mapper; hardware attempt followed by CPU fallback; DWPose sessions fall back as a pair |

The required `SessionLoader` trait separates provider setup from application I/O.
SceneWorks implements platform-library preparation and commits exclusively through
its existing `model_io::ort_session` external-data/progress seam. There is no raw
`commit_from_file` or `commit_from_memory` in this crate. The consumer's typed load
errors round-trip through `Error::Load`; engine validation and cancellation retain
separate variants. Consumers still own weight paths, downloads and caches.

26 numeric/state-transition tests moved from the worker or cover the newly shared
fallback owner. All are registered as ordinary safe controls. They execute no
model, ORT session or learned layer. The existing CPU and Mac provider CI lanes
compile/lint this crate and run those exact controls. Provider correctness remains
an explicit, separately dispatched SceneWorks `cv-provider-validation.yml` proof
against the final clean pair; the consumer runbook lists models, exact cases,
provider execution records and source/fixture provenance. No performance or memory
calibration is implied by extraction or compilation.
