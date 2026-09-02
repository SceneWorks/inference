//! Standalone SC-20676 real-model evidence harness.

fn main() -> std::process::ExitCode {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    match mlx_llm::sc20676_evidence::sc20676_cli(&args) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("sc20676-packed-evidence: {error}");
            std::process::ExitCode::FAILURE
        }
    }
}
