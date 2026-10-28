// Phone engineering evidence excludes trainer and host-fixture feature combinations.
#[doc(hidden)]
pub const IOS_ENGINEERING_FEATURES_SUPPORTED: bool = mlx_gen::IOS_ENGINEERING_FEATURES_SUPPORTED
    && mlx_gen_pid::IOS_ENGINEERING_FEATURES_SUPPORTED
    && mlx_gen_wan::IOS_ENGINEERING_FEATURES_SUPPORTED;
