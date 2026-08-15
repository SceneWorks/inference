# Vendored `mmaudio` — the frozen MMAudio torch-parity oracle (sc-17285)

`mmaudio/` is a **verbatim copy** of the reference PyTorch implementation from

    https://github.com/hkchengrex/MMAudio @ 974010a026c731054592d8f777218bd9d85a6c24
    (2026-02-23T00:09:17-06:00 — "mobile friendly")
    subtree `mmaudio/`

It is the ground truth that `scripts/reference/mmaudio_reference.py` runs to produce the five
committed torch-parity fixtures under `../tests/fixtures/`, which the native candle MMAudio port is
gated against. It is **read-only**: nothing in this directory is imported by any Rust crate, shipped
in any bundle, or on any product path — it exists so the oracle survives.

## Why it is committed

**Vendoring a parity oracle is a license decision, and this one clears it:** `hkchengrex/MMAudio`
is **MIT** (`mmaudio/LICENSE`, copied verbatim from the upstream repository root — Copyright (c)
2024 Sony Research Inc.), which permits redistribution provided the copyright and permission notice
travel with the copy. They do, both at the package root and in the three nested notices upstream
already ships (`ext/bigvgan/LICENSE`, `ext/bigvgan_v2/LICENSE`, `ext/synchformer/LICENSE`, plus the
`incl_licenses/` directories beside them). MIT into this Apache-2.0 repository is the same call
epic 14034 made for Mage-Flow in `crates/media/mlx-gen/_vendor/mage_flow`.
`../NOTICE` records the attribution.

The reason to take that option here rather than fetch-at-regeneration is that the reference is a
**package**, not a file. `mmaudio_reference.py` drives `MMAudio`, `FlowMatching`, `FeaturesUtils`,
`AutoEncoderModule`, `Synchformer` and both BigVGAN generations across 102 files; checksum-asserting
a fetch of that surface (the `moss_audio_codec_reference.py` pattern, which covers three files) is
strictly worse than committing it. It is also 364 KB across 102 files, so the cost is negligible.

## What was and was not copied

| path | vendored? | note |
| --- | --- | --- |
| `mmaudio/**` | yes | the entire package, verbatim — 102 files, incl. the nested BigVGAN/Synchformer licences and the `bigvgan_vocoder.yml` / `divided_224_16x4.yaml` configs the loaders read |
| `LICENSE` | yes | upstream repo-root MIT, copied in as `mmaudio/LICENSE` beside the code it licenses |
| `config/`, `training/`, `docs/`, `demo.py`, `train.py`, `batch_eval.py`, … | **no** | repository-level entry points and training configs; the producer builds the models directly and reads none of them |

The whole package is taken rather than the reachable subset deliberately: an import-driven subset
has to be re-derived every time upstream moves an import, and getting it wrong fails as a confusing
`ModuleNotFoundError` inside a half-loaded reference. Everything vendored is **byte-for-byte
upstream** — no local patches. Verify with:

```sh
git -C /path/to/MMAudio checkout 974010a026c731054592d8f777218bd9d85a6c24
diff -r --exclude=__pycache__ --exclude=LICENSE \
  /path/to/MMAudio/mmaudio crates/audio/candle-audio-mmaudio/_vendor/mmaudio
```

(`LICENSE` is excluded because it lives at the upstream **repo root**, not inside `mmaudio/`; it was
copied in beside the code it licenses, so it is the one file that exists only here. With that single
exclude the diff is empty — anything else it prints is a local patch and a bug.)

The harness deliberately does **not** edit the vendored source to run offline. Two hub lookups are
rebound from `scripts/reference/mmaudio_reference.py` instead, so the `diff -r` above stays empty:

* `FeaturesUtils.__init__` would build the CLIP tower via
  `create_model_from_pretrained('hf-hub:apple/DFN5B-CLIP-ViT-H-14-384')`. That literal is a
  historical upstream alias and remains untouched here so this tree stays byte-for-byte vendored;
  the canonical repository is `apple/DFN5B-CLIP-ViT-H-14-378`. MMAudio nevertheless feeds 384px,
  and patch 14 with stride 14 produces the same 27×27 grid at native 378px and at 384px. The harness
  constructs `FeaturesUtils(enable_conditions=False)` and attaches a tower built from the explicit
  canonical snapshot's own `open_clip_config.json` + `open_clip_pytorch_model.bin`, then applies the
  reference's own `patch_clip`.
* `AutoEncoderModule` would build the 44.1 kHz vocoder via
  `BigVGANv2.from_pretrained('nvidia/bigvgan_v2_44khz_128band_512x')`. The harness rebinds that
  classmethod to read the explicit snapshot directory, which upstream's `_from_pretrained` already
  supports (`os.path.isdir(model_id)`).

Both exist because inference never self-fetches and never derives a Hugging Face cache location
(epic 13657) — and that discipline has to hold for the thing generating the oracle too, or the
fixture stops being pinned to the revision the manifest pins. Keep future adaptations in the
harness, not here.

## The producer environment

The fixtures record the reference's *mathematics*, so they are generated in **float32 on the CPU**,
not the reference's own bfloat16/CUDA default. `mmaudio_parity_metadata.json` records the exact
package versions of the run that produced them, and `scripts/tests/test_mmaudio_reference.py`
re-digests this tree on every ordinary-CI run: edit anything under `mmaudio/` and the committed
fixtures are reported as stale until they are regenerated.

Regeneration, and the licence position of the fixtures themselves, are documented in
`scripts/reference/mmaudio_reference.py` and `../NOTICE`.

### SHA-256 of the vendored files

```
157b6af8c1bc4eac646abdf17473934a85e8815ade3a346cd83a58a171586a73  mmaudio/LICENSE
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  mmaudio/__init__.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  mmaudio/data/__init__.py
403e5eb01d12d204668a55adca937a2a7250501eab6a347e182b11eb20f34456  mmaudio/data/av_utils.py
8bc97374f117307782ab8a221d8c576f29ddace45b71ecf9f0aa68926b1b31b5  mmaudio/data/data_setup.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  mmaudio/data/eval/__init__.py
771c5f0a5f705054f226d2a044854937e151c759dd519f52eea3af77022c98a8  mmaudio/data/eval/audiocaps.py
ebde6a8c518b18b2c7b8a3a4379f2cf011eb2b6f171a233571cba5fa04e09cb1  mmaudio/data/eval/moviegen.py
8638c59f565844e032f08128e4f20eea3e20eeaff9e74d73d3329cb28daf5c8e  mmaudio/data/eval/video_dataset.py
432b345ff9a9307d987b2bae73a32079162df510301a4b878e59426b6de16aa0  mmaudio/data/extracted_audio.py
6db0586af00574f0784bb2aa3dae04f41f21f28407ab2ef8f022e6f778c44c46  mmaudio/data/extracted_vgg.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  mmaudio/data/extraction/__init__.py
c6ddbe0be5335d10176a4992b314bcd1922936425ce90e6840dfa4e3829c59ee  mmaudio/data/extraction/vgg_sound.py
93f9a1b25a8b421c5ef82dae9089c68dbb13801be23f48ba9168557641d98290  mmaudio/data/extraction/wav_dataset.py
7a42fc87ddf809b6a9ef6102cb8955a2c52edad2626424ae7ac66ac24b7b20a6  mmaudio/data/mm_dataset.py
b869bb0f41b7f621d2c36f39bb7947793a1c86d3a067f7dc3017d78c94d10c38  mmaudio/data/utils.py
2ef5bd7f3ff5d7ce9600a59a783a54c74348961ceefc1b53492a9e698a14586f  mmaudio/eval_utils.py
01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b  mmaudio/ext/__init__.py
429abde414b7f5bf799f666b378340695af33136682831404b8f7ce3e080f042  mmaudio/ext/autoencoder/__init__.py
c54a84bb3d7e0cd11bac2d187e30cfa070f9989f49a36dee9e674492e8da3d70  mmaudio/ext/autoencoder/autoencoder.py
838e77d4252a4771e50b204c60bf44048a5c4d37a6b694b348dc8ab6f7a5f048  mmaudio/ext/autoencoder/edm2_utils.py
ebc9d03d388e9cbbf815dfbaf53b4f11ba5e2a402729a3d1fbf87bd313fc2231  mmaudio/ext/autoencoder/vae.py
a993a05a14f95bdfc86a35f0a398d715dba90bb0d5858bad2f84780c1a20fc80  mmaudio/ext/autoencoder/vae_modules.py
da9858d516047d82096d01c112a61bd67f26d289039464d668a1d45f91738ecc  mmaudio/ext/bigvgan/LICENSE
36d064aaaff4c5654b55991617ae8d0fa6d55493a95da9ba3d02fbc1827eb9b8  mmaudio/ext/bigvgan/__init__.py
3ba94028aebabfc994bcd746bf9cbe92ecace528434c922c480be6ada182cad6  mmaudio/ext/bigvgan/activations.py
a520b10a7e52fa817b46af9cb0b8d1cf7672d3016e3632b7296cc06c11b76da0  mmaudio/ext/bigvgan/alias_free_torch/__init__.py
fb2c0ccfda50344e82227e53cab40ea6c157e9a7c33c0d8191245ba0e3807dac  mmaudio/ext/bigvgan/alias_free_torch/act.py
41890f2dc9ad822e679f17db34eb7beb7c0f5f2a3b01229c2d55393cd0538991  mmaudio/ext/bigvgan/alias_free_torch/filter.py
f9d28fb23311ecfac2b9fc7f3e16be0e6b5e8cc33f60c2e40b435f1078edcfe0  mmaudio/ext/bigvgan/alias_free_torch/resample.py
f4dd9c548d5a166525573c8f84e73e8a53858af40e2baeffc8999a7295dff8b5  mmaudio/ext/bigvgan/bigvgan.py
6b6b2ccc1b236d901195efa4341df9912a79ae08e24c348f0bb8142beefefbd1  mmaudio/ext/bigvgan/bigvgan_vocoder.yml
54ae665797fbb20ed3fc9b856be688f6b8a903b0bed8ff7edabff81dd9cdcd33  mmaudio/ext/bigvgan/env.py
5eec017ce442989e01a46cd9cd6aeb13cd0aa3b3943f681577bda00475afaa23  mmaudio/ext/bigvgan/incl_licenses/LICENSE_1
7441474b0fe03def0efc41a6340b4818359453c7cb32423c647839f8799e68df  mmaudio/ext/bigvgan/incl_licenses/LICENSE_2
7339fa418c9ad3e8e12e74ad0fd26a9cc4be8703f9c110728a992b193be85cb2  mmaudio/ext/bigvgan/incl_licenses/LICENSE_3
9a2fbb788a9d584a2a0766e4bb790c8368911143472d257952581455313cf2c6  mmaudio/ext/bigvgan/incl_licenses/LICENSE_4
b6234401b9831a86505fbeecad803a0b2fe17b06997f614d5325c6d0e1a6fa7f  mmaudio/ext/bigvgan/incl_licenses/LICENSE_5
d91bdc68d8ab6e023be9d8356e59e6556f2250832ac737e2d01d0c8b9eabf0b2  mmaudio/ext/bigvgan/models.py
23c12cd2a92babe447f9b74a1f24e68d9a341855ebcbbefdab95120d6dda96db  mmaudio/ext/bigvgan/utils.py
5c7f573db5f807a9adc2a755c4901e203ea067f73c7a20fa6b703da7e77d7b35  mmaudio/ext/bigvgan_v2/LICENSE
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  mmaudio/ext/bigvgan_v2/__init__.py
ff2562e116399bca730929aeb07e029a61321660a4f07c3db0b8f9853c70470f  mmaudio/ext/bigvgan_v2/activations.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  mmaudio/ext/bigvgan_v2/alias_free_activation/cuda/__init__.py
54778a4308d359cac8348bc28f5407652982b22732c3c768a6721183c26f5b0e  mmaudio/ext/bigvgan_v2/alias_free_activation/cuda/activation1d.py
222ce6cd687fdc1d541a4bedbf5f3578f321bc2649996b2747ebc60ddd2e1d8e  mmaudio/ext/bigvgan_v2/alias_free_activation/cuda/anti_alias_activation.cpp
e3fa60fbf80fd95cad6f05dd605c85393b6539d6684502a56b64c348c655de7d  mmaudio/ext/bigvgan_v2/alias_free_activation/cuda/anti_alias_activation_cuda.cu
39e530d6d9cf5eda60c25b899cf0ba87c70cdca3424de7cb1716adad8f212388  mmaudio/ext/bigvgan_v2/alias_free_activation/cuda/compat.h
6ee5cbfaedc6b73cf2bff9677f18bcc5a46c91a088523e853bca12805a909e8a  mmaudio/ext/bigvgan_v2/alias_free_activation/cuda/load.py
3e32f2fec72b2b6749389acd64c336b2932a3764b686634c696af8151df0e38c  mmaudio/ext/bigvgan_v2/alias_free_activation/cuda/type_shim.h
2e3138e1052e377ba2e51ea59c7d5d255a519559001757af21dbec4cb9c22471  mmaudio/ext/bigvgan_v2/alias_free_activation/torch/__init__.py
47ef38c3ca52461033a8859f4bb894bad9fa2fefafc0bc226b24e4638d8e6da3  mmaudio/ext/bigvgan_v2/alias_free_activation/torch/act.py
acf2257276e617dd3161e53abc0a1582e586a3d2d618a43235c7f83818b4e179  mmaudio/ext/bigvgan_v2/alias_free_activation/torch/filter.py
6e540908934e2f5df7c9189449165b0e2fb1e79db567ea883692e4246a32de3b  mmaudio/ext/bigvgan_v2/alias_free_activation/torch/resample.py
8b147803f8184c2cf6eedace27867d46d20b23fce31556da19c99d849a2e32e4  mmaudio/ext/bigvgan_v2/bigvgan.py
54ae665797fbb20ed3fc9b856be688f6b8a903b0bed8ff7edabff81dd9cdcd33  mmaudio/ext/bigvgan_v2/env.py
5eec017ce442989e01a46cd9cd6aeb13cd0aa3b3943f681577bda00475afaa23  mmaudio/ext/bigvgan_v2/incl_licenses/LICENSE_1
7441474b0fe03def0efc41a6340b4818359453c7cb32423c647839f8799e68df  mmaudio/ext/bigvgan_v2/incl_licenses/LICENSE_2
7339fa418c9ad3e8e12e74ad0fd26a9cc4be8703f9c110728a992b193be85cb2  mmaudio/ext/bigvgan_v2/incl_licenses/LICENSE_3
9a2fbb788a9d584a2a0766e4bb790c8368911143472d257952581455313cf2c6  mmaudio/ext/bigvgan_v2/incl_licenses/LICENSE_4
b6234401b9831a86505fbeecad803a0b2fe17b06997f614d5325c6d0e1a6fa7f  mmaudio/ext/bigvgan_v2/incl_licenses/LICENSE_5
3fb4aef5b76dc0ccb1f57499d75c2fe039df5c9793778ede172f5ac7a4045681  mmaudio/ext/bigvgan_v2/incl_licenses/LICENSE_6
8f36ba0c214d4f487ffd7ee0b8528da9b67b459bef2d115de9cd001316589f46  mmaudio/ext/bigvgan_v2/incl_licenses/LICENSE_7
ff474afe15f36bab44711fc45c1a77314a4b9d8c96fa2d333f3fefe719fd76f9  mmaudio/ext/bigvgan_v2/incl_licenses/LICENSE_8
cfd3d2e280120d48f5c87f3d81c1d8cee070da1f8f219c752abb88066b7be120  mmaudio/ext/bigvgan_v2/utils.py
892a64fee5ff50488fec6336f25d8614ff6d77c70e82d192bce93cdd8aba1d90  mmaudio/ext/mel_converter.py
66d2bc397b8d85ec77700dcbcdc62fd9079c4c595e513dc1817b83169147efca  mmaudio/ext/rotary_embeddings.py
89de5456a4fbc3cd0c1d6a2af6e7a8b7ade38764bc0c5835fbcbad6472e7714f  mmaudio/ext/stft_converter.py
c9328a125ac22561f7e322ba860613a6d9890a97436d8bdfc4f3804361c6b259  mmaudio/ext/stft_converter_mel.py
768c302e92c9c1b9c829b73f83101edbe312ca6ed3abc2e8cc9d2eb3e4b185b2  mmaudio/ext/synchformer/LICENSE
3e0a5e12b4e51fd8ba3f7e180e88a8c995f3fc99136a43621747f2cca04f8e62  mmaudio/ext/synchformer/__init__.py
80461bd19c1f7bc1213445eba48aa69554e4b0230c57d3acc80625d2e2519dce  mmaudio/ext/synchformer/divided_224_16x4.yaml
e7c5cb68dff0f95ae4343f29705044266895ac9f2ae25f79720d2a15a677e132  mmaudio/ext/synchformer/motionformer.py
d0a162d9e1ee4e215631dae93afa997867004d80b57472e6195aca6a0e6aa370  mmaudio/ext/synchformer/synchformer.py
7f5aa03e995ee57bd29c42968b9f8863fc098af56c9bf5b4ff71803d8ce21266  mmaudio/ext/synchformer/utils.py
eff9225dd341572bc7089a0c127c48d2f80bcf88cfe56f442d71057f0be0988b  mmaudio/ext/synchformer/video_model_builder.py
cbfcea6b2b74b89ea0ad2e7f657e54a942f7c8167f3324d7d707dac810a83573  mmaudio/ext/synchformer/vit_helper.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  mmaudio/model/__init__.py
258cf556c7a13cd19dd51756372dcea000125245e2c1edef5d7b615e0b527add  mmaudio/model/embeddings.py
fb91770584229d6b76000dd63a319d67d624cace27ad739fb1eb5e70a856b276  mmaudio/model/flow_matching.py
7f6720816dc544754c4bcb4ac044b2e8827f1478a3d59f48dd5e68adcea84746  mmaudio/model/low_level.py
e4c44a2c3d493b5ded26d687360f08572be797203c201437c8f5dffbd12a9ba5  mmaudio/model/networks.py
7618bb604a5500f7f3d2242dceeb68993224e37e9f75d66c542ae44e5bf161a3  mmaudio/model/sequence_config.py
2fe6d1931dee3a463dd016bc4dabe0764dff6833f381028f109bf7878449f987  mmaudio/model/transformer_layers.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  mmaudio/model/utils/__init__.py
4781b5981d4268ea99b21a667c9cab4f29a79173f30b7485a141dea32767b9f0  mmaudio/model/utils/distributions.py
cb9ab419a49a9f1d4173e45958f75a2372d71a9810e6979fbebb00f598b18dba  mmaudio/model/utils/features_utils.py
296043ef908adfc4a60ac450c4375cbf6586af9b08c88ceeed1ac9dedfda9872  mmaudio/model/utils/parameter_groups.py
873dedd0d39e74bbb77dcebd8091fad8f88351df7b01195d94472292acb620f3  mmaudio/model/utils/sample_utils.py
fdfb0192509eaedb3d80d36928ca04a0a24dd50e7f988ce26290e86c930752db  mmaudio/runner.py
699a6e5197967450f6182483cf8eaeed8171cd88d38d9676f371837b8527e2a4  mmaudio/sample.py
e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855  mmaudio/utils/__init__.py
33d910589bf87405163ef117e239342a7c5c8ec944d37040b1b9dd817ff07e4d  mmaudio/utils/dist_utils.py
ba011e378f38fd50c0818217217e52fbd5c9065600ea5463551ad0184d54f4d4  mmaudio/utils/download_utils.py
0ec4c137a39aa4c1c531d567e2e106c6952c1c7a82a5078039b995154854f212  mmaudio/utils/email_utils.py
188e1214f2ee6f7f2d80c73fb1567c49fa324ff9763f65fb68600e1a9065d390  mmaudio/utils/log_integrator.py
b4b459fb61287ae21302081c981adca7ed8eb88f49606bd4119e67dcc0b34b0c  mmaudio/utils/logger.py
3ca0e9924bed6cb62b0c999bc60ac7e82da6151fa1c623bf94bf71a1ee8bc49a  mmaudio/utils/synthesize_ema.py
889fc8d4f41bb623cec39272d409cb8ebeb175cc82b68818594e27a00cb8a439  mmaudio/utils/tensor_utils.py
567d8722697e9eaf2fe86852b6e041d0eb6139700e135cf56f1cd8543648b932  mmaudio/utils/time_estimator.py
ecf7018055c641e9892439dcde410ece6cef6fd6c0836ed7ba40fe850c1c0a76  mmaudio/utils/timezone.py
975b2b48903b09d77259069eeba36dfcdca5fb83855cbd9e954515d89d165b74  mmaudio/utils/video_joiner.py
```
