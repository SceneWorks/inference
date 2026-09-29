//! Standalone SC-20671 campaign parent/child executable.
//!
//! It deliberately has no default model path or synthetic receipt mode.  The parent consumes the
//! checked-in matrix from `mlx_llm::campaign`, forks one product worker per coordinate, and makes
//! the complete collection visible only after every child returns a sealed receipt set. An
//! operator stop file halts the parent between rows with exit status 75 (resumable).

fn main() -> std::process::ExitCode {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    match mlx_llm::campaign::sc20671_cli(&args) {
        Ok(outcome) => std::process::ExitCode::from(outcome.exit_code()),
        Err(error) => {
            eprintln!("sc20671-kv-baseline: {error}");
            std::process::ExitCode::from(2)
        }
    }
}
