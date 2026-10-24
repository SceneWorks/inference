//! Make Metal work fatal in this process (sc-25690): every [`super::cpu_process`] child installs
//! this before running its body, so a CPU-pinned test that reaches the GPU anyway — an operator
//! that ignores the CPU default, an explicit `Stream::gpu()`, a load-time GPU probe — aborts with
//! the culprit's backtrace instead of silently running on (and contending for) the GPU.
//!
//! How: the Objective-C methods every MLX GPU evaluation goes through — creating a compute or
//! blit command encoder on a command buffer (where a kernel is encoded) and `commit` (where a
//! buffer is submitted) — are replaced, on the system device's command-buffer class, with a hook
//! that reports and aborts. The hook never calls the original, so nothing reaches the GPU. MLX
//! creates command buffers eagerly per stream, which is harmless: only encoding and submission
//! are trapped.

use std::ffi::{c_char, c_void, CStr};
use std::sync::OnceLock;

type Id = *mut c_void;

#[link(name = "objc")]
extern "C" {
    fn sel_registerName(name: *const c_char) -> Id;
    fn object_getClass(obj: Id) -> Id;
    fn class_replaceMethod(
        cls: Id,
        sel: Id,
        imp: *const c_void,
        types: *const c_char,
    ) -> *const c_void;
    fn objc_msgSend();
}

#[link(name = "Metal", kind = "framework")]
extern "C" {
    fn MTLCreateSystemDefaultDevice() -> Id;
}

extern "C" {
    fn pthread_self() -> *mut c_void;
    fn pthread_getname_np(thread: *mut c_void, name: *mut c_char, len: usize) -> i32;
}

/// `[obj sel]` for a selector that takes no argument and returns an object.
unsafe fn send(obj: Id, sel: &CStr) -> Id {
    let f: extern "C" fn(Id, Id) -> Id = std::mem::transmute(objc_msgSend as *const ());
    f(obj, sel_registerName(sel.as_ptr()))
}

fn trap(what: &str) -> ! {
    // `std::thread::current()` is unusable during thread-local teardown (where MLX's encoder
    // destructor commits), so the thread name — libtest's test path — comes from pthreads.
    let mut buf = [0u8; 256];
    unsafe { pthread_getname_np(pthread_self(), buf.as_mut_ptr().cast(), buf.len()) };
    let thread = CStr::from_bytes_until_nul(&buf)
        .map(|c| c.to_string_lossy().into_owned())
        .unwrap_or_default();
    let report = format!(
        "METAL-GUARD: Metal {what} in pid {} (thread {thread:?}) — this process must not use \
         the GPU\n{}\n",
        std::process::id(),
        std::backtrace::Backtrace::force_capture()
    );
    // Straight to fd 2: libtest's output capture would swallow `eprintln!` across the abort.
    let _ = std::io::Write::write_all(&mut std::io::stderr(), report.as_bytes());
    std::process::abort();
}

extern "C" fn commit_hook(_this: Id, _sel: Id) {
    trap("command-buffer commit")
}

extern "C" fn encoder_hook(_this: Id, _sel: Id) -> Id {
    trap("command-encoder creation")
}

extern "C" fn encoder_arg_hook(_this: Id, _sel: Id, _arg: usize) -> Id {
    trap("command-encoder creation")
}

unsafe fn hook(cls: Id) {
    let replace = |sel: &CStr, imp: *const c_void, types: &CStr| {
        class_replaceMethod(cls, sel_registerName(sel.as_ptr()), imp, types.as_ptr());
    };
    replace(c"commit", commit_hook as *const c_void, c"v@:");
    replace(
        c"computeCommandEncoder",
        encoder_hook as *const c_void,
        c"@@:",
    );
    replace(c"blitCommandEncoder", encoder_hook as *const c_void, c"@@:");
    for sel in [
        c"computeCommandEncoderWithDispatchType:",
        c"computeCommandEncoderWithDescriptor:",
        c"blitCommandEncoderWithDescriptor:",
    ] {
        replace(sel, encoder_arg_hook as *const c_void, c"@@:@");
    }
}

static INSTALLED: OnceLock<bool> = OnceLock::new();

/// Trap all further Metal encoding and submission in this process (idempotent). Returns whether
/// the trap is in place (`false` only on a host with no Metal device, where there is no GPU to
/// protect).
pub fn forbid_metal() -> bool {
    let installed = *INSTALLED.get_or_init(|| unsafe {
        let device = MTLCreateSystemDefaultDevice();
        if device.is_null() {
            return false;
        }
        let queue = send(device, c"newCommandQueue");
        let mut classes = Vec::new();
        for make in [c"commandBufferWithUnretainedReferences", c"commandBuffer"] {
            let cls = object_getClass(send(queue, make));
            if !classes.contains(&cls) {
                hook(cls);
                classes.push(cls);
            }
        }
        true
    });
    if installed {
        record_model_inference_device();
    }
    installed
}

/// Opt-in proof evidence of the selected execution stream INSIDE the actual
/// model/device scope. Call before restoring that scope, never in a root
/// orchestrator on behalf of its CPU workers. No graph evaluation is added.
/// Ordinary tests (without MODEL_INFERENCE_CASE_FQN) perform no MLX query here.
pub fn record_model_inference_device() {
    record_model_inference_device_inner(metal_forbidden());
}

fn record_model_inference_device_inner(cpu_trapped: bool) {
    let Some(case) = std::env::var_os("MODEL_INFERENCE_CASE_FQN") else {
        return;
    };
    let case = case.into_string().expect("model-proof case must be UTF-8");
    assert!(
        !case.is_empty()
            && case
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b':'),
        "model-proof case must be an exact Rust test FQN"
    );
    let stream = mlx_rs::Stream::task_local_or_default();
    let device = stream
        .get_device()
        .expect("read selected model stream device");
    let selected_cpu = match device.get_type().expect("read selected model device type") {
        mlx_rs::DeviceType::Cpu => true,
        mlx_rs::DeviceType::Gpu => false,
    };
    if cpu_trapped {
        assert!(
            metal_forbidden(),
            "CPU device evidence requires the armed Metal trap"
        );
        let default = mlx_rs::Device::try_default().expect("read actual default model device");
        assert!(
            matches!(
                default.get_type().expect("read default device type"),
                mlx_rs::DeviceType::Cpu
            ),
            "CPU proof must bind the CPU default BEFORE arming the Metal trap"
        );
        assert!(
            selected_cpu,
            "CPU proof selected a non-CPU execution stream"
        );
        let default_stream = mlx_rs::Stream::cpu();
        let default_stream_device = default_stream
            .get_device()
            .expect("read default CPU stream device");
        assert!(matches!(
            default_stream_device
                .get_type()
                .expect("read default CPU stream type"),
            mlx_rs::DeviceType::Cpu
        ));
    }
    // Only the selected execution stream is primary evidence. A default device
    // or unrelated registered stream is NOT unioned into the model's device set.
    let backend = if selected_cpu { "mlx-cpu" } else { "mlx-metal" };
    eprintln!("\nMODEL_INFERENCE_DEVICE {case} {backend}");
}

/// Whether [`forbid_metal`] has armed the trap in this process.
pub fn metal_forbidden() -> bool {
    INSTALLED.get().copied().unwrap_or(false)
}
