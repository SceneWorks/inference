//! The `MageFlow` NR-MMDiT denoiser — **owned by sc-14040**.
//!
//! Port of `_vendor/mage_flow/models/mage_flow.py:58-153`. Assembly, in order:
//!
//! ```text
//! ms_pe   = MageFlowEmbedRope(img_shapes)              // crate::rope_embedder
//! img     = Linear(in_channels → hidden_size)(img)     // img_in — NO latent scale/shift
//! txt     = RMSNorm(context_in_dim, eps=1e-6)(txt)     // txt_norm
//! temb    = MageFlowTimestepProjEmbeddings(sigma)      // crate::timestep_embedder
//! txt     = Linear(context_in_dim → hidden_size)(txt)  // txt_in
//! temb    = temb + 0                                   // pooled text vec is zeroed (:116)
//! for block in transformer_blocks: (txt, img) = block(...)   // crate::transformer_block
//! img     = AdaLayerNormContinuous(img, temb); img = proj_out(img)   // crate::final_layer
//! ```
//!
//! Shapes come from [`crate::config::MageFlowConfig`], which reads **only** the nine
//! `transformer/config.json` fields the reference reads and pins the rest as constants — see that
//! module's "config-strip trap" note before adding a field here.
//!
//! The output is the flow-matching **velocity**; the sampler ([`crate::pipeline`]) applies
//! `x += (σ_next − σ_cur) · v`.
//!
//! ## Weight loading
//!
//! There is no shared `loader.rs` (see the decision note in `lib.rs`): `load_transformer` belongs
//! in **this file**, beside the module it constructs, so the concurrent text-encoder and VAE ports
//! never touch a file this story owns. The transformer is a single
//! `transformer/diffusion_pytorch_model.safetensors`, bf16, loaded non-strict (`pipeline.py:748`).
