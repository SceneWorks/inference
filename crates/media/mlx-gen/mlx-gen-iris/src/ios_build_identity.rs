// Phone engineering evidence excludes trainer and host-fixture feature combinations.
#[doc(hidden)]
pub const IOS_ENGINEERING_FEATURES_SUPPORTED: bool = !cfg!(any(feature = "training"))
    && mlx_gen::IOS_ENGINEERING_FEATURES_SUPPORTED
    && mlx_llm::IOS_ENGINEERING_FEATURES_SUPPORTED;

/// Crate-owned anchor for this compiled Cargo dependency unit and its resolved feature graph.
#[doc(hidden)]
pub struct IosCompiledDependencyAnchor;
