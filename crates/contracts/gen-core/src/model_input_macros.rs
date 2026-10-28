#[cfg(feature = "model-inference-proof")]
#[macro_export]
macro_rules! __model_input_choose {
    ($original:expr, $observed:expr) => {
        ($observed)
    };
}
#[cfg(not(feature = "model-inference-proof"))]
#[macro_export]
macro_rules! __model_input_choose {
    ($original:expr, $observed:expr) => {
        ($original)
    };
}

#[cfg(feature = "model-inference-proof")]
#[macro_export]
macro_rules! __model_input_result {
    ($original:expr, $reader:expr, $role:ident, $paths:expr) => {
        ($original).inspect(|_| {
            $crate::model_input_proof::record_reader_paths(
                $reader,
                $crate::model_input_proof::InputRole::$role,
                $paths,
            );
        })
    };
}
#[cfg(not(feature = "model-inference-proof"))]
#[macro_export]
macro_rules! __model_input_result {
    ($original:expr, $reader:expr, $role:ident, $paths:expr) => {
        ($original)
    };
}

#[cfg(feature = "model-inference-proof")]
#[macro_export]
macro_rules! __model_input_value_reader {
    ($original:expr, $reader:expr, $role:ident, $path:expr, $read:expr, origin = $origin:expr) => {
        $crate::model_input_proof::observe_path_value_tagged(
            $reader,
            $crate::model_input_proof::InputRole::$role,
            $path,
            $read,
            $origin,
        )
    };
}
#[cfg(not(feature = "model-inference-proof"))]
#[macro_export]
macro_rules! __model_input_value_reader {
    ($original:expr, $reader:expr, $role:ident, $path:expr, $read:expr, origin = $origin:expr) => {
        ($original)
    };
}

// The unselected branch is not expanded or typechecked. Ordinary builds retain
// the exact original reader expression and its original argument lifetimes.
#[cfg(feature = "model-inference-proof")]
#[macro_export]
macro_rules! __model_input_reader {
    ($original:expr, $reader:expr, $role:ident, $path:expr, $read:expr, origin = $origin:expr, when = $record_if:expr) => {
        $crate::model_input_proof::observe_path_reader_tagged(
            $reader,
            $crate::model_input_proof::InputRole::$role,
            $path,
            $read,
            $origin,
            $record_if,
        )
    };
    ($original:expr, $reader:expr, $role:ident, $path:expr, $read:expr, origin = $origin:expr) => {
        $crate::model_input_proof::observe_path_reader_tagged(
            $reader,
            $crate::model_input_proof::InputRole::$role,
            $path,
            $read,
            $origin,
            |_| true,
        )
    };
    ($original:expr, $reader:expr, $role:ident, $path:expr, $read:expr, $record_if:expr) => {
        $crate::model_input_proof::observe_path_reader_if(
            $reader,
            $crate::model_input_proof::InputRole::$role,
            $path,
            $read,
            $record_if,
        )
    };
    ($original:expr, $reader:expr, $role:ident, $path:expr, $read:expr) => {
        $crate::model_input_proof::observe_path_reader(
            $reader,
            $crate::model_input_proof::InputRole::$role,
            $path,
            $read,
        )
    };
}
#[cfg(not(feature = "model-inference-proof"))]
#[macro_export]
macro_rules! __model_input_reader {
    ($original:expr, $reader:expr, $role:ident, $path:expr, $read:expr, origin = $origin:expr, when = $record_if:expr) => {
        ($original)
    };
    ($original:expr, $reader:expr, $role:ident, $path:expr, $read:expr, origin = $origin:expr) => {
        ($original)
    };
    ($original:expr, $reader:expr, $role:ident, $path:expr, $read:expr, $record_if:expr) => {
        ($original)
    };
    ($original:expr, $reader:expr, $role:ident, $path:expr, $read:expr) => {
        ($original)
    };
}
