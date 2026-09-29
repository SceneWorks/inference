//! Standalone SC-20676 real-model evidence harness. An operator stop file halts the parent between
//! arms with exit status 75 (resumable with the same resume directory).

fn main() -> std::process::ExitCode {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    match mlx_llm::sc20676_evidence::sc20676_cli(&args) {
        Ok(outcome) => std::process::ExitCode::from(outcome.exit_code()),
        Err(error) => {
            eprintln!("sc20676-packed-evidence: {error}");
            std::process::ExitCode::FAILURE
        }
    }
}
