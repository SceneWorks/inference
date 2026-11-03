//! Shared ORT provider selection and errors. The caller owns the final commit so
//! external-data validation and load-progress instrumentation cannot be bypassed.
use crate::{Error, Result};
use ort::session::{builder::SessionBuilder, Session};
use std::path::Path;

/// Required loader callback, including application-specific external-data handling.
pub trait SessionLoader {
    /// Prepare platform dynamic-library lookup before provider registration.
    /// The application owns deployment paths and environment configuration.
    fn prepare_accelerator(&mut self) {}
    fn load(&mut self, builder: SessionBuilder, path: &Path) -> Result<Session>;
}

#[cfg(target_os = "macos")]
pub const ACCEL_DEVICE: &str = "coreml";
#[cfg(not(target_os = "macos"))]
pub const ACCEL_DEVICE: &str = "cuda";

pub(crate) fn ort_err<R>(error: ort::Error<R>) -> Error {
    Error::from(error)
}

fn build_session(path: &Path, accel: bool, load: &mut dyn SessionLoader) -> Result<Session> {
    let mut builder = Session::builder().map_err(ort_err)?;
    if accel {
        load.prepare_accelerator();
        #[cfg(target_os = "macos")]
        {
            // Retain the shipped CoreML registration policy; CUDA alone requires
            // explicit provider failure to distinguish its CPU fallback.
            builder = builder
                .with_execution_providers([ort::ep::CoreML::default().build()])
                .map_err(ort_err)?;
        }
        #[cfg(not(target_os = "macos"))]
        {
            builder = builder
                .with_execution_providers([ort::ep::CUDA::default().build().error_on_failure()])
                .map_err(ort_err)?;
        }
    }
    load.load(builder, path)
}

/// Load a model or a group of models on the same provider. A failed hardware
/// load drops the partial group before rebuilding every session on CPU. In
/// particular, DWPose must never cache a mixed hardware/CPU pair.
pub(crate) fn load_sessions<const N: usize>(
    paths: [&Path; N],
    label: &str,
    load: &mut dyn SessionLoader,
) -> Result<([Session; N], &'static str)> {
    load_group(paths, label, |path, accel| build_session(path, accel, load))
}

// The provider-independent state transition is fixture-testable without creating
// an ORT environment or executing a model.
fn load_group<T, const N: usize>(
    paths: [&Path; N],
    label: &str,
    mut load: impl FnMut(&Path, bool) -> Result<T>,
) -> Result<([T; N], &'static str)> {
    let mut build = |accel| -> Result<[T; N]> {
        let mut sessions = Vec::with_capacity(N);
        for path in paths {
            sessions.push(load(path, accel)?);
        }
        sessions
            .try_into()
            .map_err(|_| Error::Engine("session group size mismatch".into()))
    };
    match build(true) {
        Ok(sessions) => Ok((sessions, ACCEL_DEVICE)),
        Err(error) => {
            tracing::warn!(%error, provider = ACCEL_DEVICE, model = label,
                "CV hardware session build failed; falling back to CPU");
            Ok((build(false)?, "cpu"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn failed_second_model_discards_the_hardware_pair_before_cpu_retry() {
        use std::cell::Cell;
        let active = Cell::new(0);
        struct SessionToken<'a>(&'a Cell<usize>);
        impl Drop for SessionToken<'_> {
            fn drop(&mut self) {
                self.0.set(self.0.get() - 1);
            }
        }
        let mut calls = Vec::new();
        let (sessions, device) = load_group(
            [Path::new("det"), Path::new("pose")],
            "fixture",
            |path, accel| {
                calls.push((path.to_owned(), accel));
                if accel && path == Path::new("pose") {
                    return Err(Error::Engine("EP failed".into()));
                }
                if !accel && path == Path::new("det") {
                    assert_eq!(active.get(), 0);
                }
                active.set(active.get() + 1);
                Ok(SessionToken(&active))
            },
        )
        .unwrap();
        assert_eq!(device, "cpu");
        assert_eq!(
            calls,
            [
                ("det".into(), true),
                ("pose".into(), true),
                ("det".into(), false),
                ("pose".into(), false)
            ]
        );
        drop(sessions);
        assert_eq!(active.get(), 0);
    }

    #[test]
    fn failed_first_model_skips_second_hardware_load_and_preserves_cpu_load_error() {
        let mut calls = Vec::new();
        let result = load_group::<(), 2>(
            [Path::new("det"), Path::new("pose")],
            "fixture",
            |path, accel| {
                calls.push((path.to_owned(), accel));
                if accel {
                    Err(Error::Engine("EP failed".into()))
                } else {
                    Err(Error::InvalidPayload("external data missing".into()))
                }
            },
        );
        assert!(
            matches!(result, Err(Error::InvalidPayload(message)) if message == "external data missing")
        );
        assert_eq!(calls, [("det".into(), true), ("det".into(), false)]);
    }

    #[test]
    fn successful_hardware_load_does_not_rebuild_on_cpu() {
        let (sessions, device) = load_group([Path::new("model")], "fixture", |_, accel| {
            assert!(accel);
            Ok(42)
        })
        .unwrap();
        assert_eq!(sessions, [42]);
        assert_eq!(device, ACCEL_DEVICE);
    }
}
