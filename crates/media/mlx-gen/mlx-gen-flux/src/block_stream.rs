//! Exact snapshot-backed materialization of FLUX.1's joint and single DiT stacks.
//!
//! The non-block trunk stays on [`crate::transformer::FluxTransformer`]. A deferred transformer
//! owns only this reopenable description: every window opens the already-verified canonical
//! `transformer/model.safetensors`, reconstructs the requested blocks, drains the tensor handles
//! read by their constructors, and verifies the complete behavioral inventory around the lazy MLX
//! materialization boundary.

use mlx_gen::adapters::AdaptableHost;
use mlx_gen::weights::Weights;
use mlx_gen::{Error, Result};

use crate::artifact_inventory::{PackedArtifactInventory, PinnedArtifact};
use crate::transformer::{JointBlock, SingleBlock};

pub(crate) fn ensure_prepacked(host: &mut impl AdaptableHost, what: &str) -> Result<()> {
    for path in host.adaptable_paths() {
        let segments = path.split('.').collect::<Vec<_>>();
        let packed = host
            .adaptable_mut(&segments)
            .is_some_and(|linear| linear.is_quantized());
        if !packed {
            return Err(Error::Unsupported(format!(
                "flux1 block stream: {what} linear `{path}` is dense under a packed Q4/Q8 inventory"
            )));
        }
    }
    Ok(())
}

#[derive(Clone)]
pub(crate) struct FluxBlockStream {
    inventory: PackedArtifactInventory,
    source: PinnedArtifact,
    joint_blocks: usize,
    single_blocks: usize,
    quant_bits: Option<i32>,
}

impl FluxBlockStream {
    pub(crate) fn new(
        inventory: PackedArtifactInventory,
        joint_blocks: usize,
        single_blocks: usize,
        quant_bits: Option<i32>,
    ) -> Self {
        let source = inventory.transformer_source().clone();
        Self {
            inventory,
            source,
            joint_blocks,
            single_blocks,
            quant_bits,
        }
    }

    pub(crate) fn joint_blocks(&self) -> usize {
        self.joint_blocks
    }

    pub(crate) fn single_blocks(&self) -> usize {
        self.single_blocks
    }

    fn verify_inventory(&self) -> Result<()> {
        self.inventory
            .ensure_unchanged()
            .map_err(|error| Error::Msg(error.to_string()))?;
        self.source
            .ensure_unchanged()
            .map_err(|error| Error::Msg(error.to_string()))
    }

    /// Open the exact pinned single-file transformer view. No directory scan, glob, or shard
    /// resolution is permitted after contract admission.
    pub(crate) fn open(&self) -> Result<Weights> {
        self.verify_inventory()?;
        let view = Weights::from_file(self.source.canonical_path())?;
        self.verify_inventory()?;
        Ok(view)
    }

    pub(crate) fn materialize_joint(&self, view: &mut Weights, index: usize) -> Result<JointBlock> {
        if index >= self.joint_blocks {
            return Err(Error::Msg(format!(
                "flux1 block stream: joint block {index} is outside the {}-block stack",
                self.joint_blocks
            )));
        }
        let mut block = JointBlock::from_weights(view, &format!("transformer_blocks.{index}"))?;
        // Array handles are refcounted. Drain precisely what this constructor read so dropping the
        // completed window can release its materialized weights instead of retaining a map copy.
        view.remove_accessed();
        if let Some(bits) = self.quant_bits {
            // Packed Q4/Q8 constructors auto-detect `{path}.scales`; quantize is intentionally a
            // no-op for those packed leaves. Reject a lying marker rather than quantizing per
            // window, which would violate rung 4's device-format-transfer cost contract.
            ensure_prepacked(&mut block, &format!("joint block {index}"))?;
            block.quantize(bits)?;
        }
        Ok(block)
    }

    pub(crate) fn materialize_single(
        &self,
        view: &mut Weights,
        index: usize,
    ) -> Result<SingleBlock> {
        if index >= self.single_blocks {
            return Err(Error::Msg(format!(
                "flux1 block stream: single block {index} is outside the {}-block stack",
                self.single_blocks
            )));
        }
        let mut block =
            SingleBlock::from_weights(view, &format!("single_transformer_blocks.{index}"))?;
        view.remove_accessed();
        if let Some(bits) = self.quant_bits {
            ensure_prepacked(&mut block, &format!("single block {index}"))?;
            block.quantize(bits)?;
        }
        Ok(block)
    }

    /// Called only after the carried activations have been evaluated. MLX can otherwise retain a
    /// lazy reference to a replaced source until after the window has nominally completed.
    pub(crate) fn verify_materialized_window(&self) -> Result<()> {
        self.verify_inventory()
    }
}

#[derive(Clone, Copy)]
pub(crate) struct FluxBlockWindow<'a> {
    pub(crate) joint: mlx_gen::block_residency::BlockPlan,
    pub(crate) single: mlx_gen::block_residency::BlockPlan,
    pub(crate) cancel: &'a mlx_gen::CancelFlag,
    pub(crate) calibration_stream_fault: bool,
}

impl<'a> FluxBlockWindow<'a> {
    pub(crate) fn new(
        joint_blocks: usize,
        single_blocks: usize,
        size: usize,
        cancel: &'a mlx_gen::CancelFlag,
        calibration_stream_fault: bool,
    ) -> Result<Self> {
        Ok(Self {
            joint: mlx_gen::block_residency::BlockPlan::new(joint_blocks, size)?,
            single: mlx_gen::block_residency::BlockPlan::new(single_blocks, size)?,
            cancel,
            calibration_stream_fault,
        })
    }
}

pub(crate) fn evict_resident_blocks<Joint, Single>(
    joint: &mut Vec<Joint>,
    single: &mut Vec<Single>,
    expected_joint: usize,
    expected_single: usize,
) -> Result<()> {
    if joint.len() != expected_joint || single.len() != expected_single {
        return Err(Error::Msg(format!(
            "flux1: cannot finalize joint/single stream {expected_joint}/{expected_single} from resident stacks {}/{}",
            joint.len(),
            single.len()
        )));
    }
    joint.clear();
    single.clear();
    if !joint.is_empty() || !single.is_empty() {
        return Err(Error::Msg(
            "flux1: deferred transformer retained resident blocks".to_owned(),
        ));
    }
    Ok(())
}

/// Deterministic calibration-only failure point. The caller invokes this immediately after joint
/// block zero has been reconstructed, while the shared window driver still owns the active view.
pub(crate) fn calibration_stream_fault(enabled: bool, joint_index: usize) -> Result<()> {
    if enabled && joint_index == 0 {
        return Err(Error::Msg(
            "flux1: calibration fault after joint block materialization".to_owned(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_gen::gen_core::block_window::{BlockPlan, BlockWindowBackend};
    use std::cell::Cell;

    #[derive(Default)]
    struct FakeBackend {
        opens: usize,
        releases: Cell<usize>,
    }

    impl BlockWindowBackend for FakeBackend {
        type View = ();

        fn open_view(&mut self) -> mlx_gen::gen_core::Result<Self::View> {
            self.opens += 1;
            Ok(())
        }

        fn release(&self) {
            self.releases.set(self.releases.get() + 1);
        }
    }

    #[test]
    fn calibration_fault_is_first_joint_only_and_a_fresh_probe_succeeds() {
        assert!(calibration_stream_fault(true, 0).is_err());
        assert!(calibration_stream_fault(true, 1).is_ok());
        assert!(calibration_stream_fault(false, 0).is_ok());

        // Exercise the exact shared scheduling/error boundary without constructing MLX arrays. The
        // first run fails after its simulated block reconstruction; a fresh run over the same plan
        // traverses every block, proving the fault has no retained provider state.
        let plan = BlockPlan::new(2, 1).unwrap();
        let cancel = mlx_gen::CancelFlag::default();
        let mut failing = FakeBackend::default();
        let error = mlx_gen::gen_core::block_window::run_windowed(
            &mut failing,
            &plan,
            &cancel,
            0usize,
            |count, (), range| {
                for index in range {
                    calibration_stream_fault(true, index)
                        .map_err(mlx_gen::gen_core::Error::from)?;
                }
                Ok(count + 1)
            },
            |_| Ok(()),
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("after joint block materialization"));
        assert_eq!(failing.opens, 1);
        assert_eq!(failing.releases.get(), 1);

        let mut recovery = FakeBackend::default();
        let completed = mlx_gen::gen_core::block_window::run_windowed(
            &mut recovery,
            &plan,
            &cancel,
            0usize,
            |count, (), _range| Ok(count + 1),
            |_| Ok(()),
        )
        .unwrap();
        assert_eq!(completed, 2);
        assert_eq!(recovery.opens, 2);
        assert_eq!(recovery.releases.get(), 2);
    }

    #[test]
    fn exact_joint_and_single_plans_are_independent_and_evict_both_stacks() {
        let mut joint = vec![0_u8; 19];
        let mut single = vec![0_u8; 38];
        evict_resident_blocks(&mut joint, &mut single, 19, 38).unwrap();
        assert!(joint.is_empty() && single.is_empty());

        let cancel = mlx_gen::CancelFlag::default();
        let window = FluxBlockWindow::new(19, 38, 1, &cancel, false).unwrap();
        assert_eq!(window.joint.n_blocks(), 19);
        assert_eq!(window.joint.window_count(), 19);
        assert_eq!(window.single.n_blocks(), 38);
        assert_eq!(window.single.window_count(), 38);

        let canceled = mlx_gen::CancelFlag::default();
        canceled.cancel();
        let canceled_window = FluxBlockWindow::new(19, 38, 1, &canceled, false).unwrap();
        let mut backend = FakeBackend::default();
        let error = mlx_gen::gen_core::block_window::run_windowed(
            &mut backend,
            &canceled_window.joint,
            canceled_window.cancel,
            (),
            |(), (), _range| Ok(()),
            |_| Ok(()),
        )
        .unwrap_err();
        assert!(matches!(error, mlx_gen::gen_core::Error::Canceled));
        assert_eq!(
            backend.opens, 0,
            "pre-cancel must not open the pinned stream"
        );
        assert_eq!(
            backend.releases.get(),
            1,
            "a pre-canceled window must still clear the backend allocator"
        );

        let mut wrong_joint = vec![0_u8; 18];
        let mut exact_single = vec![0_u8; 38];
        assert!(evict_resident_blocks(&mut wrong_joint, &mut exact_single, 19, 38).is_err());
        assert_eq!(
            wrong_joint.len(),
            18,
            "a rejected shape must not partially evict"
        );
        assert_eq!(exact_single.len(), 38);
    }
}
