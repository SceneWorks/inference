//! sc-25690: the Metal trap of the CPU children is live. Every `cpu_process` body runs with
//! `common::metal_guard` armed, and a GPU op in a process that armed it aborts before any kernel
//! is encoded — so a CPU-pinned test that reaches the GPU fails loudly instead of passing on it.

use mlx_rs::{Array, StreamOrDevice};

use crate::common::{cpu_process, metal_guard};

#[test]
fn cpu_process_bodies_run_with_metal_forbidden() {
    cpu_process(
        "cpu_child_guard::cpu_process_bodies_run_with_metal_forbidden",
        || {
            assert!(
                metal_guard::metal_forbidden(),
                "the CPU child armed no trap"
            );
            // CPU work is unaffected.
            let a = Array::from_slice(&[1f32, -2.0], &[2]);
            assert_eq!(a.abs().unwrap().sum(None).unwrap().item::<f32>(), 3.0);
        },
    );
}

const PROBE: &str = "IRIS_METAL_GUARD_PROBE";

#[test]
fn a_gpu_op_after_forbid_metal_aborts_the_process() {
    if std::env::var_os(PROBE).is_some() {
        assert!(metal_guard::forbid_metal());
        let a = Array::from_slice(&[1f32, -2.0], &[2]);
        let _ = a.abs_device(StreamOrDevice::gpu()).unwrap().eval();
        return; // reached only when the trap failed — the parent then sees a clean exit
    }
    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "cpu_child_guard::a_gpu_op_after_forbid_metal_aborts_the_process",
            "--nocapture",
            "--test-threads",
            "1",
        ])
        .env(PROBE, "1")
        .output()
        .expect("spawn the probe");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(!out.status.success(), "the GPU op ran:\n{stderr}");
    assert!(stderr.contains("METAL-GUARD: Metal"), "{stderr}");
}
