// Phone engineering evidence excludes trainer and host-fixture feature combinations.
#[doc(hidden)]
pub const IOS_ENGINEERING_FEATURES_SUPPORTED: bool =
    !cfg!(any(feature = "training")) && mlx_gen::IOS_ENGINEERING_FEATURES_SUPPORTED;
