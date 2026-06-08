//! # mlx-gen-joycaption
//!
//! JoyCaption provider crate for [`mlx-gen`](mlx_gen). Linking this crate registers the
//! `fancyfeast/llama-joycaption-beta-one-hf-llava` captioner with the core caption registry.

pub mod model;

pub use model::{descriptor, load, load_joycaption, JoyCaption};
