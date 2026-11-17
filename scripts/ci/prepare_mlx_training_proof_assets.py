#!/usr/bin/env python3
"""Bind provisioned, pinned snapshots for the bounded MLX training proofs (SW-7172).

The explicit model-proof workflow binds no per-family snapshot environment for MLX rows on the
provisioned Mac (nax-macos-2), so each bounded MLX training proof below names the exact pinned
Hub snapshot(s) its test reads, and this setup resolves each one under the runner-local model
roots, in order: `$MLX_GEN_MODELS_ROOT` (the repository variable, `~` expanded), the provisioned
`/Volumes/Models/huggingface/hub`, and the runner-local `~/sceneworks-rw-weights/hub`. A root that
is (or sits under) a symlink is never used. The resolved paths are appended to `GITHUB_ENV`.

This reads file metadata only: the production readers inside the ONE manual invocation still
authenticate/load the actual inputs; a path is not runtime proof. When a pin is absent the setup
FAILS with an inventory of every revision each pinned repository holds under each root (and which
required files each revision has), so the first dispatch shows what the Mac really holds. It never
lists a Hub root itself (a full Hub listing blocked forever on nax-macos-2); only the bounded
`models--org--name/snapshots` directories, each under a timeout.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

PROVISIONED_HUB = Path("/Volumes/Models/huggingface/hub")
LISTING_TIMEOUT_SECONDS = 20.0


@dataclass(frozen=True)
class Pin:
    """One pinned snapshot directory: `<root>/models--org--name/snapshots/<revision>/<subdir>`."""

    repository: str
    revision: str
    subdir: str = ""
    files: tuple[str, ...] = ()
    # `root`: the test resolves the snapshot itself under this models root (the binding is the root).
    binds_root: bool = False

    def relative(self) -> Path:
        path = Path("models--" + self.repository.replace("/", "--")) / "snapshots" / self.revision
        return path / self.subdir if self.subdir else path


def proof(package: str, fqn: str, target: str = "integration") -> str:
    return f"{package}/{target}/{fqn}"


# Candle-side pins where the repository matches (prepare_cohort/bespoke_training_proof_assets.py);
# MLX-converted repositories at the revisions the MLX real-weight lanes and code reference.
KOLORS = Pin("Kwai-Kolors/Kolors-diffusers", "7e091c75199e910a26cd1b51ed52c28de5db3711",
             files=("model_index.json", "unet/config.json", "vae/config.json", "text_encoder/config.json",
                    "tokenizer/tokenizer.json"))
LENS = Pin("SceneWorks/Lens", "5c5521d4417a3cae55816929ece69319d1e7712a",
           files=("model_index.json", "transformer/lens_bf16.safetensors", "text_encoder/config.json",
                  "tokenizer/tokenizer.json", "vae/config.json"))
SDXL = Pin("stabilityai/stable-diffusion-xl-base-1.0", "462165984030d82259a11f4367a4eed129e94a7b",
           files=("model_index.json", "unet/config.json", "vae/config.json", "text_encoder/config.json",
                  "text_encoder_2/config.json", "tokenizer/vocab.json", "tokenizer/merges.txt"))
Z_IMAGE = Pin("Tongyi-MAI/Z-Image-Turbo", "f332072aa78be7aecdf3ee76d5c247082da564a6",
              files=("model_index.json", "transformer/config.json", "text_encoder/config.json",
                     "tokenizer/tokenizer.json", "vae/config.json"))
MAGE_BASE = Pin("SceneWorks/Mage-Flow-Base", "d642341926fcdb450c17c8fbda03759e8f731c9b", "bf16",
                files=("transformer/config.json",))
LTX = Pin("SceneWorks/ltx-2.3-mlx", "254989c3ca7ee691187647f350b112c0c448789d", "q8", files=("config.json",))
LTX_GEMMA = Pin("SceneWorks/ltx-2.3-mlx", "254989c3ca7ee691187647f350b112c0c448789d", "gemma",
                files=("config.json", "tokenizer.json"))
WAN_TI2V_5B = Pin("SceneWorks/wan2.2-ti2v-5b-mlx", "bb1b055249614cf9d7cf4373fbdbc184b77dee88", "bf16",
                  files=("config.json", "model.safetensors", "t5_encoder.safetensors", "tokenizer.json",
                         "vae.safetensors"))
ANIMA = Pin("circlestone-labs/Anima", "53eec3898af698b2cf2a11379021fc9c5465d228", "split_files",
            files=("diffusion_models/anima-base-v1.0.safetensors", "text_encoders/qwen_3_06b_base.safetensors",
                   "vae/qwen_image_vae.safetensors"), binds_root=True)
KREA_RAW = Pin("krea/Krea-2-Raw", "4ad9f4b627a647fad78b3dfeebb09f2654aeb494",
               files=("model_index.json", "transformer/config.json", "text_encoder/config.json",
                      "tokenizer/tokenizer.json", "vae/config.json"))
SD35_MEDIUM = Pin("stabilityai/stable-diffusion-3.5-medium", "b940f670f0eda2d07fbb75229e779da1ad11eb80",
                  files=("model_index.json", "transformer/config.json", "vae/config.json",
                         "text_encoder/config.json", "text_encoder_3/config.json"))
QWEN_IMAGE_2_1 = Pin("Qwen/Qwen-Image-2.1", "790c92633540aa0cb11d9abf19eb46d861714758",
                     files=("model_index.json", "transformer/config.json", "vae/config.json",
                            "text_encoder/config.json"))
MOSS_BASE = Pin("OpenMOSS-Team/MOSS-Image-Base", "1ceede80badddd6a7812069619edaa5633587192",
                files=("transformer/config.json", "transformer/diffusion_pytorch_model.safetensors.index.json"))
MOSS_SHARED = Pin("OpenMOSS-Team/MOSS-Image-SFT", "fc5de0d56fc910cbca6f6625a70f5f582b26deb1",
                  files=("text_encoder/config.json", "tokenizer/tokenizer.json", "vae/config.json"))

# Exact registered identities (`package/target/fqn`) -> {environment variable: pinned snapshot}.
MLX_TRAINING_PROOFS: dict[str, dict[str, Pin]] = {
    proof("mlx-gen-kolors", "trainer_e2e::kolors_trainer_emits_preview_samples"): {"KOLORS_SNAPSHOT": KOLORS},
    proof("mlx-gen-lens", "trainer_e2e::lens_trainer_emits_preview_samples"): {"LENS_SNAPSHOT": LENS},
    proof("mlx-gen-sdxl", "trainer_e2e::sdxl_trainer_emits_preview_samples"): {"SDXL_SNAPSHOT": SDXL},
    proof("mlx-gen-z-image", "trainer_e2e::z_image_trainer_emits_preview_samples"): {"ZIMAGE_SNAPSHOT": Z_IMAGE},
    proof("mlx-gen-mage", "trainer_e2e::mage_trainer_emits_preview_samples"): {"MAGE_BASE_SNAPSHOT": MAGE_BASE},
    proof("mlx-gen-ltx", "trainer_e2e::ltx_trainer_emits_preview_samples"): {
        "LTX_BASE_DIR": LTX, "LTX_GEMMA_DIR": LTX_GEMMA},
    proof("mlx-gen-wan", "trainer_e2e::wan_ti2v_5b_trainer_emits_preview_samples"): {
        "WAN_TI2V_5B_MODEL_DIR": WAN_TI2V_5B},
    proof("mlx-gen-anima", "training::train_emits_in_training_previews"): {"MLX_GEN_MODELS_ROOT": ANIMA},
    proof("mlx-gen-krea", "trainer_real_weights::short_train_produces_loadable_adapter"): {"KREA_RAW_DIR": KREA_RAW},
    proof("mlx-gen-sd3", "training::real_weight_repro::medium_training_smoke_round_trip", "mlx_gen_sd3"): {
        "SD3_MEDIUM_DIR": SD35_MEDIUM},
    proof("mlx-gen-qwen-image-2-1", "lora_real_weights::t2i_lora_trains_a_portrait_and_previews_at_its_canvas"): {
        "MLX_GEN_QWEN_IMAGE_2_1_SNAPSHOT": QWEN_IMAGE_2_1},
    proof("mlx-gen-moss-image", "real_weights::moss_bounded_training_trains_a_portrait"): {
        "MOSS_BASE_SNAPSHOT": MOSS_BASE, "MOSS_SHARED_SNAPSHOT": MOSS_SHARED},
}


# Repositories reported by a failure inventory beside the pinned ones: plausible alternative homes
# of the same weights (upstream vs SceneWorks re-host, raw vs MLX-converted), so one failed dispatch
# shows every candidate the Mac holds.
INVENTORY_ALSO = ("microsoft/Mage-Flow-Base", "SceneWorks/Mage-Flow-Components-mlx", "SceneWorks/Lens-Turbo",
                  "SceneWorks/sd3.5-medium-mlx", "SceneWorks/z-image-turbo-mlx", "SceneWorks/krea-2-turbo-mlx",
                  "Wan-AI/Wan2.2-TI2V-5B-Diffusers", "SceneWorks/qwen-image-2-1-mlx")


def exact_bindings(rows: list[dict], backend: str, runner: str | None) -> dict[str, Pin] | None:
    """The pinned bindings of ONE exact registered MLX training proof on the provisioned Mac."""
    if backend != "mlx" or runner != "nax" or len(rows) != 1:
        return None
    row = rows[0]
    if row.get("target_kind") not in ("test", "lib") or "selected_configuration" in row:
        return None
    return MLX_TRAINING_PROOFS.get(row["package"] + "/" + row["target"] + "/" + row["fqn"])


def candidate_roots(env: dict[str, str], home: Path) -> list[tuple[str, Path]]:
    """The ordered runner-local model roots; a configured `~` root expands against this runner."""
    roots = []
    configured = env.get("MLX_GEN_MODELS_ROOT", "")
    if configured:
        expanded = home / configured[2:] if configured.startswith("~/") else Path(configured)
        if os.path.isabs(expanded):
            roots.append(("MLX_GEN_MODELS_ROOT", expanded))
    roots.append(("provisioned", PROVISIONED_HUB))
    roots.append(("runner-local", home / "sceneworks-rw-weights" / "hub"))
    unique, seen = [], set()
    for label, root in roots:
        if root not in seen:
            seen.add(root)
            unique.append((label, root))
    return unique


def linked_root(root: Path) -> bool:
    """A root that is, or sits under, a symlink is never a binding source."""
    return any(part.is_symlink() for part in (root, *root.parents) if part != part.parent)


def complete(path: Path, pin: Pin) -> list[str]:
    """The pin's required files missing or empty under `path`."""
    missing = []
    for relative in pin.files:
        file = path / relative
        if not file.is_file() or file.stat().st_size == 0:
            missing.append(relative)
    return missing


def resolve(pin: Pin, roots: list[tuple[str, Path]]) -> Path | None:
    """The first non-linked root holding the complete pinned snapshot."""
    for _, root in roots:
        if linked_root(root) or not root.is_dir():
            continue
        path = root / pin.relative()
        if path.is_dir() and not complete(path, pin):
            return root if pin.binds_root else path
    return None


def bounded_listing(directory: Path) -> list[str] | str:
    """`directory`'s entry names, or why it could not be listed within the timeout."""
    result: dict[str, object] = {}

    def work() -> None:
        try:
            result["names"] = sorted(entry.name for entry in os.scandir(directory))
        except OSError as error:
            result["error"] = f"{type(error).__name__}: {error}"

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    thread.join(LISTING_TIMEOUT_SECONDS)
    if thread.is_alive():
        return f"listing timed out after {LISTING_TIMEOUT_SECONDS:.0f}s"
    return result.get("names", result.get("error", "unlisted"))  # type: ignore[return-value]


def inventory(pins: list[Pin], roots: list[tuple[str, Path]], also: tuple[str, ...] = ()) -> dict:
    """For each pinned (or alternative) repository: the revisions each root holds; for a pinned
    revision layout, which required files each revision is missing, else its top-level entries."""
    report: dict[str, dict] = {}
    repositories = list(dict.fromkeys([pin.repository for pin in pins] + list(also)))
    for repository in repositories:
        same = [pin for pin in pins if pin.repository == repository]
        entry = report.setdefault(repository, {
            "pinned": [{"revision": p.revision, "subdir": p.subdir, "files": list(p.files)} for p in same],
            "roots": {}})
        for label, root in roots:
            key = f"{label}:{root}"
            if linked_root(root):
                entry["roots"][key] = "root is (under) a symlink; never used"
                continue
            if not root.is_dir():
                entry["roots"][key] = "root absent"
                continue
            snapshots = root / ("models--" + repository.replace("/", "--")) / "snapshots"
            if not snapshots.is_dir():
                entry["roots"][key] = "repository absent"
                continue
            listed = bounded_listing(snapshots)
            if isinstance(listed, str):
                entry["roots"][key] = listed
                continue
            revisions: dict[str, object] = {}
            for revision in listed:
                if same:
                    revisions[revision] = {
                        (pin.subdir or "."): ("absent" if not (snapshots / revision / pin.subdir).is_dir()
                                              else (complete(snapshots / revision / pin.subdir, pin) or "complete"))
                        for pin in same}
                else:
                    revisions[revision] = bounded_listing(snapshots / revision)
            entry["roots"][key] = revisions
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    identities = plan.get("bindings", [])
    if (plan.get("backend") != "mlx" or plan.get("selected") != 1 or len(identities) != 1
            or identities[0] not in MLX_TRAINING_PROOFS or plan.get("request", {}).get("runner") != "nax"
            or plan.get("asset_setup") != "mlx-training"):
        raise ValueError("MLX training asset setup requires one exact reviewed case on the provisioned Mac")
    env = dict(os.environ)
    if env.get("RUNNER_OS") != "macOS" or env.get("RUNNER_NAME") != "nax-macos-2":
        raise ValueError("MLX training proofs bind the provisioned secondary Mac's model roots")
    github_env = env.get("GITHUB_ENV", "")
    if not github_env:
        raise ValueError("GitHub Actions environment file is required")
    bindings = MLX_TRAINING_PROOFS[identities[0]]
    roots = candidate_roots(env, Path.home())
    resolved = {name: resolve(pin, roots) for name, pin in bindings.items()}
    missing = [name for name, path in resolved.items() if path is None]
    if missing:
        every = list({pin: None for case in MLX_TRAINING_PROOFS.values() for pin in case.values()})
        print(json.dumps({"case": identities[0], "missing": missing,
                          "roots": [f"{label}:{root}" for label, root in roots],
                          "inventory": inventory(every, roots, INVENTORY_ALSO)}, indent=2), flush=True)
        raise ValueError("pinned MLX training snapshot(s) absent from every runner-local root: " + ", ".join(missing))
    with open(github_env, "a", encoding="utf-8") as out:
        for name, path in resolved.items():
            if "\n" in str(path) or not re.fullmatch(r"[A-Z][A-Z0-9_]*", name):
                raise ValueError(f"invalid binding {name}")
            out.write(f"{name}={path}\n")
    print(json.dumps({"case": identities[0], "installed_paths": {k: str(v) for k, v in resolved.items()},
                      "claim": "metadata-only asset binding; actual production reader authentication "
                               "belongs to the one manual invocation"}), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as error:
        print("MLX training asset setup rejected: " + str(error), file=sys.stderr)
        raise SystemExit(1)
