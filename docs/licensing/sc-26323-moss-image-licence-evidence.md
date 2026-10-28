# MOSS checkpoint licence evidence (sc-26323)

Read on 2026-10-18 from the exact locally acquired frozen snapshots. The approved
acquisition manifest records full SHA-256 and file sizes; `gen_core::moss_image`
now exposes the same path-free frozen byte authority for both providers.

| Component | Primary evidence | Declaration |
| --- | --- | --- |
| `moss_image_base` | [Base LICENSE at 1ceede80](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base/blob/1ceede80badddd6a7812069619edaa5633587192/LICENSE) | Apache License 2.0 |
| `moss_image_sft` | [SFT LICENSE at fc5de0d5](https://huggingface.co/OpenMOSS-Team/MOSS-Image-SFT/blob/fc5de0d56fc910cbca6f6625a70f5f582b26deb1/LICENSE) | Apache License 2.0 |
| `moss_image_sft_8_steps_dmd` | [DMD README at 5ff314b4](https://huggingface.co/lodestones/MOSS-Image-SFT-8-steps-dmd/blob/5ff314b4c1470876b114b810146085db8a181a6e/README.md) | Front matter `license: apache-2.0` |

The Base and SFT LICENSE files have SHA-256
`cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30`.
The DMD package has no separate LICENSE; the frozen card declares its license and
SFT parent. Its README SHA-256 is
`714b1072d9bd9667f6a6615164271d0eba78aad2930d3ccefa3b5741e09bb957`.
These three rows describe the exact bundles, including their bundled dependencies;
they do not infer another checkpoint's terms from reused native Rust code. They
reuse the existing `apache-2-0` family. The provider mappings enumerate optional
DMD assets only for SFT. Metadata is disclosure and does not gate execution.

The small author transformer source preserved with synthetic fixtures is covered
by the exact SFT LICENSE copied as `tiny_author/LICENSE.author`. Its source hash
and frozen revision are recorded in the fixture receipts. The evaluated SceneWorks reference source is AGPL-3.0 and is not copied here;
regeneration takes an explicit external source path, checks its recorded hash, and
extracts the two reference AST definitions only at fixture-generation time. The
checked-in JSON contains synthetic inputs and numerical outputs.
