//! Seeded tiny-model fixtures for the CPU unit tests (test-only).
//!
//! candle's CPU backend ignores `Device::set_seed`, so `Tensor::randn` draws from the
//! process-global RNG: a fixture built with it is different on every run, and an assertion that
//! holds for almost every draw fails intermittently in CI
//! (`training::tests::depth_anchoring::aux_only_step_trains_the_adapter_through_depth_alone`
//! did). Every random fixture tensor in this crate's tests is drawn here, from a `StdRng` seeded
//! per test, so the same seed gives the same tensors on every run and platform.

use candle_gen::candle_core::{Device, Tensor};
use candle_gen::candle_nn::VarMap;
use rand::rngs::StdRng;

/// `N(0, std²)` of shape `dims`, drawn on the CPU from the seeded `rng`, then moved to `dev`.
pub(crate) fn randn_seeded(rng: &mut StdRng, std: f32, dims: &[usize], dev: &Device) -> Tensor {
    let n = dims.iter().product();
    let v: Vec<f32> = candle_gen::seeded_normal_vec(rng, n)
        .into_iter()
        .map(|x| x * std)
        .collect();
    Tensor::from_vec(v, dims, &Device::Cpu)
        .unwrap()
        .to_device(dev)
        .unwrap()
}

/// Set every var in `vm` to `N(0, std²)` from `rng`, in name order — the map is a `HashMap`, so
/// its own iteration order differs per process and would reshuffle which var gets which draw.
pub(crate) fn randomize_vars(vm: &VarMap, rng: &mut StdRng, std: f32, dev: &Device) {
    let data = vm.data().lock().unwrap();
    let mut names: Vec<&String> = data.keys().collect();
    names.sort();
    for name in names {
        let v = &data[name];
        v.set(&randn_seeded(rng, std, v.dims(), dev)).unwrap();
    }
}
