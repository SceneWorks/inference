// Cargo can enable this crate's proof feature through another workspace member.
// Select the observer only when the invoking crate also enables its proof feature:
// the observed expression can refer to source-selection state gated at that call site.
#[cfg(feature = "model-inference-proof")]
#[macro_export]
macro_rules! __model_input_choose {
    ($original:expr, $observed:expr) => {{
        #[cfg(feature = "model-inference-proof")]
        {
            $observed
        }
        #[cfg(not(feature = "model-inference-proof"))]
        {
            $original
        }
    }};
}
#[cfg(not(feature = "model-inference-proof"))]
#[macro_export]
macro_rules! __model_input_choose {
    ($original:expr, $observed:expr) => {
        ($original)
    };
}

#[macro_export]
macro_rules! __model_input_result {
    ($original:expr, $reader:expr, $role:ident, $paths:expr) => {
        $crate::__model_input_choose!(
            $original,
            ($original).inspect(|_| {
                $crate::model_input_proof::record_reader_paths(
                    $reader,
                    $crate::model_input_proof::InputRole::$role,
                    $paths,
                );
            })
        )
    };
}

#[macro_export]
macro_rules! __model_input_value_reader {
    ($original:expr, $reader:expr, $role:ident, $path:expr, $read:expr, origin = $origin:expr) => {
        $crate::__model_input_choose!(
            $original,
            $crate::model_input_proof::observe_path_value_tagged(
                $reader,
                $crate::model_input_proof::InputRole::$role,
                $path,
                $read,
                $origin,
            )
        )
    };
}

#[macro_export]
macro_rules! __model_input_reader {
    ($original:expr, $reader:expr, $role:ident, $path:expr, $read:expr, origin = $origin:expr, when = $record_if:expr) => {
        $crate::__model_input_choose!(
            $original,
            $crate::model_input_proof::observe_path_reader_tagged(
                $reader,
                $crate::model_input_proof::InputRole::$role,
                $path,
                $read,
                $origin,
                $record_if,
            )
        )
    };
    ($original:expr, $reader:expr, $role:ident, $path:expr, $read:expr, origin = $origin:expr) => {
        $crate::__model_input_choose!(
            $original,
            $crate::model_input_proof::observe_path_reader_tagged(
                $reader,
                $crate::model_input_proof::InputRole::$role,
                $path,
                $read,
                $origin,
                |_| true,
            )
        )
    };
    ($original:expr, $reader:expr, $role:ident, $path:expr, $read:expr, $record_if:expr) => {
        $crate::__model_input_choose!(
            $original,
            $crate::model_input_proof::observe_path_reader_if(
                $reader,
                $crate::model_input_proof::InputRole::$role,
                $path,
                $read,
                $record_if,
            )
        )
    };
    ($original:expr, $reader:expr, $role:ident, $path:expr, $read:expr) => {
        $crate::__model_input_choose!(
            $original,
            $crate::model_input_proof::observe_path_reader(
                $reader,
                $crate::model_input_proof::InputRole::$role,
                $path,
                $read,
            )
        )
    };
}
