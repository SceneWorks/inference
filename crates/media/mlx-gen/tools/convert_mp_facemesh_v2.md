# FaceMesh-v2 checkpoint for the face-landmark loss (epic 2123, sc-24831)

The face-landmark training loss (`mlx-gen-face` / `candle-gen-face` `train::FaceLandmarkLoss`) runs
MediaPipe's FaceMesh-v2 landmark detector natively. Its only published PyTorch form is
`py-feat/mp_facemesh_v2`, which ships two torch-specific files: a pickled onnx2torch
`GraphModule` (`face_landmarks_detector_Nx3x256x256_onnx.pth`) and a TorchScript archive
(`face_landmarks_detector.pt`). Neither loads natively. `convert_mp_facemesh_v2.py` lowers the
GraphModule to a **`sceneworks-fx-program/1`** safetensors file. The parameters are tensors and the
op list is JSON in `__metadata__.program`. The format is documented in `fx_program.py` and parsed by
`gen_core::fx_program`. Both native executors run it. The script re-executes the lowered program and
refuses to write output that differs from torch by more than 1e-4. If the graph contains an op it
does not know, it raises and names that op.

## One-time rehost (maintainer; needs the download and the HF write token)

```sh
python3 -m venv /tmp/facemesh && . /tmp/facemesh/bin/activate
pip install torch onnx2torch safetensors huggingface_hub
huggingface-cli download py-feat/mp_facemesh_v2 face_landmarks_detector_Nx3x256x256_onnx.pth \
  --revision 39eb85054cf76fe0f57b7e12d6765ae89d89f2b5 --local-dir /tmp/facemesh/src
python3 crates/media/mlx-gen/tools/convert_mp_facemesh_v2.py \
  /tmp/facemesh/src/face_landmarks_detector_Nx3x256x256_onnx.pth /tmp/facemesh/out
# prints {"ops": [...], "self_check_max_abs": <= 1e-4, "output0_shape": [2, 1, 1, 1434], ...}
huggingface-cli repo create SceneWorks/mp-facemesh-v2 --type model   # once
huggingface-cli upload SceneWorks/mp-facemesh-v2 \
  /tmp/facemesh/out/face_landmarks_detector.safetensors face_landmarks_detector.safetensors
```

After the upload, pin the commit it reports. Add a SceneWorks `componentOnly` catalog entry for
`SceneWorks/mp-facemesh-v2` at that revision, then set `supportsFaceLandmarkLoss` on the targets
whose trainer declares `techniques.face_landmark_loss`.

If the self-check reports an unsupported op, add it to `fx_program.py`,
`gen_core::fx_program::Op`, and both executors (`mlx-gen-face/src/program.rs` and
`candle-gen-face/src/program.rs`). Extend `crates/media/face_loss_fixtures/` so the new op is
pinned across torch, MLX and Candle.
