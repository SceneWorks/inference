# Boogu reference / parity harness (dev tooling)

PyTorch-on-MPS scripts that drive the **upstream** Boogu reference pipeline to (a) sanity-check
quality and (b) dump golden tensors for the MLX port's parity tests (E2/E3). These are reproduction
notes — they are **not** part of the Rust build.

## Setup
- A torch+diffusers+transformers env with MPS (this repo was validated against `~/mlx-flux-venv`:
  torch 2.12, diffusers 0.37, transformers 5.9).
- The upstream package on `PYTHONPATH`: `git clone https://github.com/boogu-project/Boogu-Image`
  then `PYTHONPATH=/path/to/Boogu-Image`.
- Weights in the HF cache: `hf download Boogu/Boogu-Image-0.1-Base` (and `-Turbo`).

Each script monkeypatches the pipeline's CUDA-only device validator and passes `device="mps"`
(the reference assumes CUDA; `enable_inner_devices_manager` stays off).

## Scripts
- `run_ref.py [h w steps]` — Base T2I (true-CFG). Writes PNGs to `reference/outputs/`.
- `run_turbo.py [h w steps]` — Turbo DMD few-step (`text_guidance_scale=1.0`, no CFG).
- `golden_dump.py` — hooks `processor.apply_chat_template` + `transformer.forward`, runs a 1-step
  256² Base gen, and writes `reference/goldens/boogu_golden.safetensors` (+ `meta.json`):
  `tok_input_ids`/`tok_attention_mask` → `instruction_hidden_states` (E2 target), plus
  `dit_in_latent_chw` / `timestep` / `dit_out_velocity_chw` (E3 targets).
